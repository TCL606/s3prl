"""
    Distiller_Fbank Model
    Author: Changli Tang (https://github.com/TCL606)
"""

from easydict import EasyDict as edict
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pretrain.distillFBANK.dataset import OnlineWaveDataset
from upstream.distiller_fbank.model import DistillerConfig, DistillerModel
import math
import wandb
import logging
import editdistance
import wandb
from argparse import Namespace

from fairseq.data.dictionary import Dictionary
from .w2l_decoder import W2lViterbiDecoder

logger = logging.getLogger(__name__)

def freeze_model(model):
    """Freeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = False


def build_decoder(dictionary_path):
    dictionary = Dictionary.load(dictionary_path)
    dec_args = Namespace()
    dec_args.nbest = 1
    dec_args.criterion = "ctc"
    dec_args.kenlm_model = '/mnt/lustre/sjtu/home/xc915/superb/nlp_utils/arpa/4-gram.mmap'
    dec_args.lexicon = '/mnt/lustre/sjtu/home/xc915/superb/nlp_utils/lexicon/librispeech_lexicon.lst'
    dec_args.beam = 50
    dec_args.beam_size_token = 32
    dec_args.beam_threshold = 32
    dec_args.lm_weight = 2.0
    dec_args.word_score = -1.0
    dec_args.unk_weight = -math.inf
    dec_args.sil_weight = 0

    return W2lViterbiDecoder(dec_args, dictionary)

class UpstreamPretrainExpert(nn.Module):
    """
    The Distiller pretrain expert
    """

    def __init__(
        self, datarc, upstream_config, device="cuda", multi_gpu=False, **kwargs
    ):
        super().__init__()

        self.datarc = datarc
        self.device = device
        self.multi_gpu = multi_gpu

        if type(upstream_config) == str:
            self.upstream_config = yaml.load(
                open(upstream_config, "r"), Loader=yaml.FullLoader
            )
            logger.info(
                "[UpstreamPretrainExpert] - Using upstream config"
            )
        elif type(upstream_config) == dict:
            self.upstream_config = upstream_config
            logger.info(
                "[UpstreamPretrainExpert] - Using upstream config from the previous experiment."
            )
        else:
            raise ValueError

        self._get_train_dataloader()

        logger.info("[UpstreamPretrainExpert] - Initializing model...")
        model_config = DistillerConfig(self.upstream_config["distiller"])
        self.model = DistillerForPretrain(
            model_config, edict(self.upstream_config["teacher"])
        ) #! inside a model has a distiller and a teacher

        if self.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
            logger.info(
                "[UpstreamPretrainExpert] - Multi-GPU training Enabled: "
                + str(torch.cuda.device_count())
            )
        logger.info(
            "[UpstreamPretrainExpert] - Number of parameters: "
            + str(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        )

    def _get_train_dataloader(self):
        dataset = OnlineWaveDataset(
            self.upstream_config["task"],
            self.datarc["train_batch_size"],
            target_level=self.upstream_config["audio"]["target_level"],
            **self.datarc,
        )

        self.dataloader = DataLoader(
            dataset,
            batch_size=1,  # for bucketing
            shuffle=True,
            num_workers=self.datarc["num_workers"],
            drop_last=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

    # Interface
    def load_model(self, all_states):
        if self.multi_gpu:
            self.model.module.distiller.load_state_dict(all_states["Distiller"])
        else:
            self.model.distiller.load_state_dict(all_states["Distiller"])

    # Interface
    def add_state_to_save(self, all_states):
        all_states["Distiller"] = (
            self.model.float().distiller.state_dict()
            if not self.multi_gpu
            else self.model.float().module.distiller.state_dict()
        )
        all_states["Config"] = self.upstream_config
        return all_states

    # Interface
    def get_train_dataloader(self):
        return self.dataloader

    # Interface
    def forward(self, data, records={}, global_step=0, log_step=1000, **kwargs): #! true forward step
        """
        Args:
            data:
                [wave_input, pad_mask]

            records:
                defaultdict(list), by appending contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records every log_step

        Return:
            loss
        """

        wave_input, wave_orig, wave_len, pad_mask = data
        wave_input = wave_input.to(self.device)
        wave_len = wave_len.to(self.device)
        pad_mask = pad_mask.type(wave_input.dtype).to(self.device)

        loss, other_res = self.model( #! DistillerForPretrain
            wave_input,
            wave_orig,
            wave_len,
            pad_mask,
            return_other=global_step % log_step == 0,
        )

        return loss, records

    # interface
    def on_before_zero_grad(self):
        pass

    # interface
    def log_records(self, records, logger, prefix, global_step, **kwargs):
        """
        Args:
            records:
                defaultdict(list), contents already appended

            logger:
                Tensorboard SummaryWriter
                please use f'{prefix}your_content_name' as key name
                to log your customized contents

            prefix:
                used to indicate downstream and train/test on Tensorboard
                eg. 'phone/train-'

            global_step:
                global_step in runner, which is helpful for Tensorboard logging
        """
        for key, values in records.items():
            if isinstance(values, torch.Tensor) and len(values.shape) > 1:
                logger.add_image(f"{prefix}{key}", values, global_step=global_step)
            elif isinstance(values, float):
                logger.add_scalar(f"{prefix}{key}", values, global_step=global_step)

class DistillerForPretrain(nn.Module):
    """
    Distiller for pretraining
    """

    def __init__(self, config: DistillerConfig, teacher_config: edict):
        super().__init__()
        self.config = config
        self.distiller = DistillerModel(config) #! build the distiller model #! what is the shape of config
        logger.info("[DistillerModel] - The structure of distiller model")
        logger.info(self.distiller)

        self.teacher_config = teacher_config
        # teacher = torch.hub.load("s3prl/s3prl", teacher_config.model) #! get the teacher model

        #! get the model locally
        if teacher_config.model.find("wav2vec2") >= 0:
            logger.info("[DistillerForPretrain] - use wav2vec2 as teacher model")
            from upstream.wav2vec2.expert import UpstreamExpert
            teacher = UpstreamExpert(teacher_config.model_path)
        elif teacher_config.model.find("data2vec") >= 0:
            logger.info("[DistillerForPretrain] - use data2vec as teacher model")
            from upstream.data2vec.expert import UpstreamExpert
            teacher = UpstreamExpert(teacher_config.model_path)
        else:
            logger.info("[DistillerForPretrain] - use hubert as teacher model")
            from upstream.hubert.expert import UpstreamExpert
            teacher = UpstreamExpert(teacher_config.model_path)
        logger.info(teacher.model)

        if (
            teacher_config.model.find("hubert") >= 0
            or teacher_config.model.find("wav2vec2") >= 0
            or teacher_config.model.find("data2vec") >= 0
        ):
            teacher.model.encoder.layerdrop = 0
            logger.info("[DistillerForPretrain] - Disabled teacher's encoder layerdrop")

        # if(teacher_config.model.find("data2vec") >= 0):
        #     logger.info(teacher.model.state_dict)
        #     teacher.model.cfg.task.normalize = False

        assert self.distiller.n_tasks <= teacher_config.n_layers, (
            self.distiller.n_tasks,
            teacher_config.n_layers,
        )
        self.teacher = teacher
        freeze_model(self.teacher)

        logger.info(
            "[DistillerForPretrain] - Using {} as teacher with {} layers".format(
                teacher_config.model, teacher_config.n_layers
            )
        )

        if config.loss_type == "l1": #! use l1 loss
            self.loss_func = nn.L1Loss(reduction="none")
            self.frontend_loss_func = nn.L1Loss(reduction="mean")
        elif config.loss_type == "l2":
            self.frontend_loss_func = nn.MSELoss(reduction="mean")
            self.loss_func = nn.MSELoss(reduction="none")
            
        else:
            raise NotImplementedError(config.loss_type)


        self.rec_loss = config.rec_loss
        if self.rec_loss > 0:
            logger.info("[DistillerForPretrain] - Enabled rec loss.")

        self.cosine_loss = config.cosine_loss #! 1.0
        if self.cosine_loss > 0:
            logger.info("[DistillerForPretrain] - Enabled cosine similarity loss.")

        self.hidden_loss = config.hidden_loss #! 1.0
        if self.hidden_loss > 0:
            logger.info("[DistillerForPretrain] - Enabled hidden loss.")

        self.attn_loss = config.attn_loss #! 1.0
        if self.attn_loss > 0:
            logger.info("[DistillerForPretrain] - Enabled attn loss.")

        self.embedding_loss = config.embedding_loss
        if self.embedding_loss > 0:
            logger.info("[DistillerForPretrain] - Enabled embedding loss.")
        
        self.frontend_loss = config.frontend_loss
        if self.frontend_loss > 0:
            logger.info("[DistillerForPretrain] - Enabled frontend loss.")
            
        self.frontend_steps = config.frontend_steps

        self.temperature = config.temperature
        self.use_temperature = config.use_temperature

        #! copy value from teacher
        if config.init_teacher_conv_layers:
            logger.info(
                "[DistillerForPretrain] - "
                "Initializing feature extractor from teacher"
            )
            self.distiller.feature_extractor.load_state_dict(
                self.teacher.model.feature_extractor.state_dict(),strict=False
            )
            if self.distiller.post_extract_proj is not None and self.distiller.post_extract_proj.weight.shape == self.teacher.model.post_extract_proj.weight.shape and self.distiller.post_extract_proj.bias.shape == self.teacher.model.post_extract_proj.bias.shape:
                self.distiller.post_extract_proj.load_state_dict(
                    self.teacher.model.post_extract_proj.state_dict()
                )

        if config.init_teacher_encoder_layers:
            logger.info("[DistillerForPretrain] - " "Initializing encoder from teacher")
            self.distiller.encoder.pos_conv.load_state_dict(
                self.teacher.model.encoder.pos_conv.state_dict(),strict=False
            )
            for l in range(config.encoder_layers): #! 2
                self.distiller.encoder.layers[l].load_state_dict(
                    self.teacher.model.encoder.layers[l].state_dict()
                )

        # freeze the label embedding
        if hasattr(self.teacher.model, 'label_embs_concat'):
            self.teacher.model.label_embs_concat.requires_grad = False

        # decode
        self.steps = 0
        self.enable_decode = config.enable_decode
        self.dictionary_path = config.dictionary_path
        if(self.enable_decode):
            self.decoder = build_decoder(self.dictionary_path)

        # final proj
        self.projection_type = config.projection_type

    def forward(
        self,
        wave_input: torch.Tensor,
        wave_orig: list,
        wave_len: torch.Tensor,
        pad_mask: torch.Tensor,
        return_other: bool = False,
    ):
        """
        Forward function.
        Input:
            wave_input: FloatTensor (B x T_wave)
            wave_orig: List of FloatTensor
            wave_len: LongTensor (B)
            pad_mask: FloatTensor (B x T)
            return_other: Bool (returns other information for logging)
        """
        return_other = False
        # Forward model
        if self.config.get_hidden == False:
            feat, feat_final, pred, pad_mask, student_embeddings = self.distiller(wave_input, wave_len, pad_mask) #! distiller model(the small one)
        else:
            feat, feat_final, pred, pad_mask, student_hiddens, student_attns = self.distiller(wave_input, wave_len, pad_mask, get_hidden=True)
            student_hiddens = torch.stack(student_hiddens, dim=1)
            student_attns = torch.stack(student_attns, dim=1)
            #! feat [12, 752, 512] BNxTxD   feat_final [12, 752, 768]  pred [12, 2, 752, 768] BxNxTxD
            #! feat: after conv  feat_final: after proj  pred: after transformers

        with torch.no_grad():
            wave_orig = [wave.to(wave_input.device) for wave in wave_orig]
            with torch.cuda.amp.autocast(False):
                teacher_hiddens = self.teacher(wave_orig)
                # get the embeddings
                teacher_logits = teacher_hiddens["hidden_states"][-1] # B x T x C
                teacher_frontend = teacher_hiddens["hidden_states"][0]
                if hasattr(self.teacher.model, 'get_embeddings'):
                    teacher_embeddings = self.teacher.model.get_embeddings(teacher_logits)
                else:
                    teacher_embeddings = None
            if self.config.task_emb_type == "none":
                teacher_hiddens = teacher_hiddens["hidden_states"][self.config.n_tasks]
                teacher_hiddens = teacher_hiddens.unsqueeze(1)
            else:
                if self.config.task_emb_type in ["expand-last", "hnet", "self-hidden", "layer-wise"]:
                    teacher_hiddens = [
                        teacher_hiddens["hidden_states"][i]
                        for i in self.distiller.pred_layer_id
                    ]
                else:
                    teacher_hiddens = teacher_hiddens["hidden_states"][1:]
                teacher_hiddens = torch.stack(teacher_hiddens, dim=1)  # B x N x T x D

        (
            total_loss,
            rec_loss,
            rec_layer_loss,
            sim_loss,
            sim_layer_loss,
            emb_loss,
            frontend_loss
        ) = self.compute_loss(feat, pred, teacher_hiddens, teacher_frontend, teacher_embeddings, student_embeddings, return_other) #! feat [12, 752, 512]  pred [12, 2, 752, 768]  teacher_hiddens [12, 2, 752, 768]

        wandb.log({
            "total_loss": total_loss,
            "rec_loss": rec_loss,
            "sim_loss": sim_loss,
            "emb_loss": emb_loss,
            "frontent_loss": frontend_loss
        })

        # for i, item in enumerate(rec_layer_loss):
        #     wandb.log({f"rec_layer_loss_{i}":item})

        # for i, item in enumerate(sim_layer_loss):
        #     wandb.log({f"sim_layer_loss_{i}":item})
        
        return_other = False

        if return_other:
            with torch.no_grad():
                other_res = {
                    "rec_loss": rec_loss,
                    "sim_loss": sim_loss,
                    "norm_feat_final": feat_final.pow(2).mean(),
                }
                teacher_norm = torch.abs(teacher_hiddens).mean((0, 2, 3))
                if self.config.task_emb_type == "none":
                    other_res[f"rec_l{self.config.n_tasks}"] = rec_layer_loss[0]
                    other_res[f"tar_norm_l{self.config.n_tasks}"] = teacher_norm[0]
                    if sim_layer_loss is not None:
                        other_res[f"sim_l{self.config.n_tasks}"] = sim_layer_loss[0]
                else:
                    for i in range(self.config.n_tasks):
                        layer_id = i + 1
                        if self.config.task_emb_type in [
                            "expand-last",
                            "hnet",
                            "self-hidden",
                        ]:
                            layer_id = self.distiller.pred_layer_id[i]
                        other_res[f"rec_l{layer_id}"] = rec_layer_loss[i]
                        other_res[f"tar_norm_l{layer_id}"] = teacher_norm[i]
                        if sim_layer_loss is not None:
                            other_res[f"sim_l{layer_id}"] = sim_layer_loss[i]
                    if self.config.task_emb_type not in [
                        "expand-last",
                        "hnet",
                        "self-hidden",
                    ]:
                        other_res[
                            "norm_task_emb"
                        ] = self.distiller.task_embedding.weight.pow(2).mean()
        else:
            other_res = None

        return total_loss, other_res

    def compute_loss(self, feat, pred, target, teacher_frontend, teacher_embeddings, student_embeddings, return_other=False):
        """
        Computes loss.
        Inputs:
            feat: B x T x D
            pred: B x N x T x D
            target: B x N x T x D
        """
        #! why feat is not same as pred in last dimension(because it is the output of conv)
        # Reconstruction loss
        assert pred.shape == target.shape, (pred.shape, target.shape)
        
        frontend_loss = 0
        rec_loss = 0
        rec_layer_loss = None
        sim_loss = 0
        sim_layer_loss = None
        emb_loss = 0
        
        if self.steps < self.frontend_steps:
            if self.frontend_loss > 0:
                frontend_loss = self.frontend_loss_func(teacher_frontend, feat)
                # frontend_loss = frontend_loss.mean()
            else:
                frontend_loss = 0
        else: 
            if self.rec_loss > 0:
                rec_loss = self.loss_func(pred, target)  # B x N x T x D #! L1 loss

                with torch.no_grad():
                    rec_layer_loss = rec_loss.mean((0, 2, 3))

                rec_loss = rec_loss.mean()
            else:
                rec_loss = 0
                rec_layer_loss = None

            # Cosine similarity loss
            if self.cosine_loss > 0:
                sim_loss = -F.logsigmoid(F.cosine_similarity(pred, target, dim=-1)) #! what is the dimension of every val? # sim_loss [12, 2, 752]
                # B x N x T
                with torch.no_grad():
                    sim_layer_loss = sim_loss.mean((0, 2))
                sim_loss = sim_loss.mean()
            else:
                sim_loss = 0
                sim_layer_loss = None

            # Feature loss
            # feat_pen = feat.float().pow(2).mean() #? what is feature loss? (maybe it is not important)

            # embedding loss
            # get pseudo label sequences
            if self.embedding_loss > 0:
                embedding_size = teacher_embeddings.shape # B x T x 256
                teacher_embeddings = teacher_embeddings.view(embedding_size[0] * embedding_size[1], embedding_size[2]) # BT x 256

                with torch.no_grad():
                    label_embs = self.teacher.model.label_embs_concat
                    label_embs_teacher = label_embs.unsqueeze(1).expand(-1, teacher_embeddings.size(0), -1) # [504, b*t, 256]
                    teacher_embedding_logits = torch.cosine_similarity(teacher_embeddings.float(), label_embs_teacher.float(), dim=-1).type_as(teacher_embeddings)
                    teacher_embedding_logits = teacher_embedding_logits.transpose(0, 1) # [b*t, 504]
                    teacher_embedding_logits =  teacher_embedding_logits / self.temperature

                if self.projection_type == 'type1':
                    student_embeddings = student_embeddings.view(embedding_size[0] * embedding_size[1], embedding_size[2]) # [b*t, 256]
                    student_embedding_logits = torch.cosine_similarity(student_embeddings.float(), label_embs_teacher.float(), dim=-1).type_as(student_embeddings)
                    student_embedding_logits = student_embedding_logits.transpose(0, 1) # [b*t, 504]
                    if self.use_temperature:
                        student_embedding_logits = student_embedding_logits / self.temperature

                elif self.projection_type == 'type2':
                    student_embedding_logits = student_embeddings.view(embedding_size[0] * embedding_size[1], student_embeddings.shape[2]) # [b*t,504]
                    if self.use_temperature:
                        student_embedding_logits = student_embedding_logits / self.temperature

                if (self.projection_type == 'type1' or self.projection_type == 'type2'):
                    teacher_prob_log = F.log_softmax(teacher_embedding_logits, dim=-1)
                    student_prob_log = F.log_softmax(student_embedding_logits, dim=-1) 
                    emb_loss = F.kl_div(input = student_prob_log,
                                        target = teacher_prob_log,
                                        log_target = True,
                                        reduction = 'batchmean')

                elif self.projection_type == 'type3':
                    with torch.no_grad():
                        teacher_prob = F.softmax(teacher_embedding_logits, dim = -1)
                        teacher_label = teacher_prob.argmax(dim=-1)
                        pos = torch.index_select(label_embs, 0, teacher_label)
                        negs = label_embs.unsqueeze(1).expand(-1, teacher_embeddings.size(0), -1)
                        neg_is_pos = (pos == negs).all(-1)
                        pos = pos.unsqueeze(0)
                        targets = torch.cat([pos, negs], dim=0)
                    student_embeddings = student_embeddings.view(embedding_size[0] * embedding_size[1], embedding_size[2])
                    student_logits = torch.cosine_similarity(student_embeddings.float(), targets.float(), dim=-1).type_as(student_embeddings)
                    if self.use_temperature:
                        student_logits /= self.temperature
                    if neg_is_pos.any():
                        student_logits[1:][neg_is_pos] = float("-inf")
                    student_logits = student_logits.transpose(0, 1) # BT x 505
                    target_list = torch.zeros(student_logits.shape[0]).long().cuda()
                    emb_loss = F.cross_entropy(student_logits, target_list, reduction='mean')

                # valid
                if (self.steps % 200 == 0 and self.enable_decode):
                    w_errs = 0
                    w_len = 0
                    logger.info(f"[Valid] - begin to valid -- step {self.steps}")
                    with torch.no_grad():
                        teacher_prob_log_t = teacher_prob_log.view(embedding_size[0], embedding_size[1], -1).float().contiguous().cpu().unsqueeze(0) # 1 x B x T x kinds
                        student_prob_log_t = student_prob_log.view(embedding_size[0], embedding_size[1], -1).float().contiguous().cpu().unsqueeze(0) # 1 x B x T x kinds
                        for (teacher_lp, student_lp) in zip(teacher_prob_log_t[0:3], student_prob_log_t[0:3]):
                            teacher_decoded = None
                            teacher_decoded = self.decoder.decode(teacher_lp)
                            if len(teacher_decoded) < 1:
                                teacher_decoded = None
                            else:
                                teacher_decoded = teacher_decoded[0]
                                if len(teacher_decoded) < 1:
                                    teacher_decoded = None
                                else:
                                    teacher_decoded = teacher_decoded[0]

                            student_decoded = None
                            student_decoded = self.decoder.decode(student_lp)
                            if len(student_decoded) < 1:
                                student_decoded = None
                            else:
                                student_decoded = student_decoded[0]
                                if len(student_decoded) < 1:
                                    student_decoded = None
                                else:
                                    student_decoded = student_decoded[0]

                            if(teacher_decoded is not None and student_decoded is not None and "tokens" in teacher_decoded and "tokens" in student_decoded):
                                teacher_words = teacher_decoded["tokens"]
                                student_words = student_decoded["tokens"]

                                teacher_string = ''
                                student_string = ''
                                for token in teacher_words:
                                    teacher_string += str(int(token))
                                    teacher_string += ' '
                                for token in student_words:
                                    student_string += str(int(token))
                                    student_string += ' '
                                logger.info(f'teacher string: {teacher_string}')
                                logger.info(f'student string: {student_string}')
                                w_errs += editdistance.eval(teacher_string.split(" "), student_string.split(" "))
                                w_len += len(teacher_string)

                        logger.info(f'wer: {w_errs/w_len}')
                        wandb.log({'wer': w_errs/w_len})
            else:
                emb_loss = 0
        self.steps += 1

        total_loss = (
            rec_loss * self.rec_loss
            + sim_loss * self.cosine_loss
            + emb_loss * self.embedding_loss
            + frontend_loss * self.frontend_loss
        )

        return total_loss, rec_loss, rec_layer_loss, sim_loss, sim_layer_loss, emb_loss, frontend_loss

    def compute_loss_hidden(self, feat, pred, target, teacher_hiddens, teacher_attns, student_hiddens, student_attns, return_other=False):
        """
        Computes loss.
        Inputs:
            feat: B x T x D
            pred: B x N x T x D
            target: B x N x T x D
            student_hidden: B x N x T x D
            student_attn: B x N x T x D
        """
        # Reconstruction loss
        assert pred.shape == target.shape, (pred.shape, target.shape)
        rec_loss = self.loss_func(pred, target)  # B x N x T x D #! L1 loss

        with torch.no_grad():
            rec_layer_loss = rec_loss.mean((0, 2, 3))

        rec_loss = rec_loss.mean()

        # Cosine similarity loss
        if self.cosine_loss > 0:
            sim_loss = -F.logsigmoid(F.cosine_similarity(pred, target, dim=-1)) #! what is the dimension of every val? # sim_loss [12, 2, 752]
            # B x N x T
            with torch.no_grad():
                sim_layer_loss = sim_loss.mean((0, 2))
            sim_loss = sim_loss.mean()
        else:
            sim_loss = 0
            sim_layer_loss = None

        # hidden loss
        assert teacher_hiddens.shape == student_hiddens.shape, (teacher_hiddens.shape, student_hiddens.shape)
        hidden_loss = self.loss_func(teacher_hiddens, student_hiddens)  # B x N x T x D #! L1 loss

        with torch.no_grad():
            hidden_layer_loss = hidden_loss.mean((0, 2, 3))
        hidden_loss = hidden_loss.mean()

        # attn loss
        attn_loss = None
        attn_layer_loss = None
        # assert teacher_attns.shape == student_attns.shape, (teacher_attns.shape, student_attns.shape)
        # attn_loss = self.loss_func(teacher_attns, student_attns)  # B x N x T x D #! L1 loss

        # if return_other:
        #     with torch.no_grad():
        #         attn_layer_loss = attn_loss.mean((0, 2, 3))
        # else:
        #     attn_layer_loss = None
        # attn_loss = attn_loss.mean()

        # Feature loss
        # feat_pen = feat.float().pow(2).mean() #? what is feature loss? (maybe it is not important)

        total_loss = 0

        total_loss = (
            rec_loss * self.rec_loss
            + sim_loss * self.cosine_loss
            + hidden_loss * self.hidden_loss
        )

        return total_loss, rec_loss, rec_layer_loss, sim_loss, sim_layer_loss, hidden_loss, hidden_layer_loss, attn_loss, attn_layer_loss
