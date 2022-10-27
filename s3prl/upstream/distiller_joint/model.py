"""
    Distiller_JOINT Model
    Author: Changli Tang (https://github.com/TCL606)
"""

import torch
from torch import nn
import logging

from .module import (
    SplitLinear,
    TransformerEncoder,
    ConvFeatureExtractionModel,
    GradMultiply,
)

import os

logger = logging.getLogger(__name__)


class DistillerConfig:
    """
    Configuration class
    """

    def __init__(self, config: dict):
        # Feature extractor
        self.extractor_mode = str(config.get("extractor_mode", "default"))
        self.extractor_conv_feature_layers = str(
            config.get(
                "extractor_conv_feature_layers",
                "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
            )
        )
        self.extractor_dropout = float(config.get("extractor_dropout", 0.0))
        self.feature_grad_mult = float(config.get("feature_grad_mult", 1.0))

        # Convolutional relative positional encoding
        self.conv_pos = int(config.get("conv_pos", 128))
        self.conv_pos_groups = int(config.get("conv_pos_groups", 16))

        # Transformer encoder
        self.encoder_layers = int(config.get("encoder_layers", 1))
        self.encoder_embed_dim = int(config.get("encoder_embed_dim", 768))
        self.encoder_ffn_embed_dim = int(config.get("encoder_ffn_embed_dim", 3072))
        self.encoder_attention_heads = int(config.get("encoder_attention_heads", 12))
        self.activation_fn = str(config.get("activation_fn", "gelu"))
        self.layer_norm_first = bool(config.get("layer_norm_first", False))
        self.attention_type = str(config.get("attention_type", "original"))

        # Dropout
        self.dropout = float(config.get("dropout", 0.1))
        self.attention_dropout = float(config.get("attention_dropout", 0.1))
        self.activation_dropout = float(config.get("activation_dropout", 0.1))
        self.encoder_layerdrop = float(config.get("encoder_layerdrop", 0.0))

        # Output
        self.final_dim = int(config.get("final_dim", 768))
        self.teacher_final_dim = int(config.get("teacher_final_dim", 768))
        self.out_layer_type = str(config.get("out_layer_type", "expand-last"))
        self.out_layer_inter_dim = int(config.get("out_layer_inter_dim", -1))

        # Task & loss
        self.n_tasks = int(config.get("n_tasks", 12))
        self.task_emb_type = str(config.get("task_emb_type", "expand-last"))
        self.task_emb_size = int(config.get("task_emb_size", 0))
        self.layer_emb_size = int(config.get("layer_emb_size", 0))
        self.loss_type = str(config.get("loss_type", "l1"))
        self.feat_pen_loss = float(config.get("feat_pen_loss", 0.0))
        self.rec_loss = float(config.get("rec_loss", 0.0))
        self.cosine_loss = float(config.get("cosine_loss", 0.0))
        self.hidden_loss = float(config.get("hidden_loss", 0.0))
        self.attn_loss = float(config.get("attn_loss", 0.0))
        self.embedding_loss = float(config.get("embedding_loss", 0.0))
        self.temperature = float(config.get("temperature", 1.0))
        self.use_temperature = bool(config.get("use_temperature", True))

        # When task_emb_type == 'expand-last' only
        self.pred_layer_id = list(
            config.get("pred_layer_id", range(1, self.n_tasks + 1))
        )
        self.pred_layer_id_2 = list(
            config.get("pred_layer_id_2", range(1, self.n_tasks + 1))
        )

        # Initialization
        self.init_model = str(
            config.get("init_model", "regression")
        )
        self.init_teacher_conv_layers = bool(
            config.get("init_teacher_conv_layers", False)
        )
        self.init_teacher_encoder_layers = bool(
            config.get("init_teacher_encoder_layers", False)
        )
        self.get_hidden = bool(
            config.get("get_hidden", False)
        )

        # final proj type
        self.projection_type = str(
            config.get("projection_type", "type1")
        )

        # decode
        self.enable_decode = bool(
            config.get("enable_decode", False)
        )
        self.dictionary_path = str(
            config.get("dictionary_path", 'dict.ltr.txt')
        )

        # picture
        self.picture_path = str(
            config.get("picture_path", '')
        )

        # kmeans
        self.kmeans_path = str(
            config.get("kmeans_path", '/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/pretrain/distiller_kmeans/kmeans/hubert_base_ls960_L9_km500.bin')
        )
        self.kmeans_layer = int(
            config.get("kmeans_layer", 12)
        )


class DistillerModel(nn.Module):
    """
    Distiller Model
    """

    def __init__(self, config: DistillerConfig):
        super().__init__()

        self.config = config

        self.conv_layers = eval(config.extractor_conv_feature_layers)
        feat_emb_dim = self.conv_layers[-1][0]
        self.feature_extractor = ConvFeatureExtractionModel(
            self.conv_layers,
            dropout=config.extractor_dropout,
            mode=config.extractor_mode,
            conv_bias=False,
        )
        self.feature_grad_mult = config.feature_grad_mult

        self.n_tasks = config.n_tasks
        self.task_emb_type = config.task_emb_type

        final_emb_size = config.encoder_embed_dim
        if self.task_emb_type == "add":
            self.task_embedding = nn.Embedding(config.n_tasks, config.encoder_embed_dim)
            nn.init.normal_(self.task_embedding.weight, 0.0, 0.1)
        elif self.task_emb_type == "concat":
            assert config.task_emb_size > 0
            feat_emb_dim += config.task_emb_size
            self.task_embedding = nn.Embedding(config.n_tasks, config.task_emb_size)
        elif self.task_emb_type == "concat-last":
            assert config.task_emb_size > 0
            self.task_embedding = nn.Embedding(config.n_tasks, config.task_emb_size)
            final_emb_size += config.task_emb_size
        elif self.task_emb_type == "expand-last":
            self.pred_layer_id = config.pred_layer_id
            self.pred_layer_id_2 = config.pred_layer_id_2
            assert self.n_tasks == len(self.pred_layer_id)
            logger.info(
                f"[DistillerModel] - Expands the output dimension by {self.n_tasks} times"
            )
            logger.info(f"[DistillerModel] - Pred layers: {self.pred_layer_id}")
            logger.info(f"[DistillerModel] - Pred hidden layers: {self.pred_layer_id_2}")
        elif self.task_emb_type == "layer-wise":
            self.pred_layer_id = config.pred_layer_id
            logger.info(
                f"[DistillerModel] - teacher model dim: {config.teacher_final_dim}, student model dim: {final_emb_size}"
            )
            logger.info(f"[DistillerModel] - Pred layers: {self.pred_layer_id}")
        elif self.task_emb_type == "self-hidden":
            self.pred_layer_id = config.pred_layer_id
            assert self.n_tasks == len(self.pred_layer_id)
            assert self.n_tasks == config.encoder_layers + 1
            logger.info("[DistillerModel] - Predicting with self-hidden layers")
            logger.info(f"[DistillerModel] - Pred layers: {self.pred_layer_id}")
        elif self.task_emb_type == "none":
            logger.info(
                f"[DistillerModel] - Disabled task embedding (predicts only layer {self.n_tasks})"
            )
        else:
            raise NotImplementedError(f"Unknown task emb type {self.task_emb_type}")

        self.post_extract_proj = (
            nn.Linear(feat_emb_dim, config.encoder_embed_dim)
            if feat_emb_dim != config.encoder_embed_dim
            else None
        )

        if config.encoder_layers > 0:
            self.encoder = TransformerEncoder(config)
        else:
            self.encoder = nn.GELU()

        final_dim = config.final_dim * (
            1 if self.task_emb_type != "expand-last" else self.n_tasks
        )

        inter_dim = config.out_layer_inter_dim
        inter_dim = inter_dim if inter_dim > 0 else final_emb_size

        logger.info(f"[DistillerModel] - Out layer type: {config.out_layer_type}")
        if config.out_layer_type == "expand-last":
            assert self.task_emb_type == "expand-last"
            logger.info(f"[DistillerModel] - Inter dim = {inter_dim}")
            self.output_layer = nn.Sequential(
                nn.Linear(final_emb_size, inter_dim * self.n_tasks),
                nn.GELU(),
                SplitLinear(inter_dim, self.n_tasks, config.final_dim),
            )
        elif config.out_layer_type in {"none", "self-hidden"}:
            self.output_layer = None
        elif config.out_layer_type == "layer-wise":
            assert self.task_emb_type == "layer-wise"
            self.output_layer = nn.ModuleList([
                nn.Linear(final_emb_size, config.teacher_final_dim) for _ in range(config.encoder_layers)
            ])
        else:
            raise NotImplementedError(f"Unknown out layer type {config.out_layer_type}")

        if config.projection_type == 'type1' or config.projection_type == 'type3':
            self.final_proj = nn.Linear(config.encoder_embed_dim, 256)
        elif config.projection_type == 'type2':
            self.final_proj = nn.Linear(config.encoder_embed_dim, 504)
        else:
            self.final_proj = nn.Linear(config.encoder_embed_dim, 500)

    def forward_feature(self, wave, pad_mask):
        """Forward feature extractor"""

        if self.feature_grad_mult > 0:
            feat = self.feature_extractor(wave)
            if self.feature_grad_mult != 1.0:
                feat = GradMultiply.apply(feat, self.feature_grad_mult)
        else:
            with torch.no_grad():
                feat = self.feature_extractor(wave)

        feat = feat.transpose(1, 2)  # B x T x D
        pad_mask = self.cal_pad_mask(pad_mask, feat.shape[1])

        return feat, pad_mask

    def forward(self, wave, pad_mask, task_id=None, get_hidden=False, no_pred=False): #! to get the output of distill model
        """
        Forward function
        Input:
            wave (FloatTensor): B x T_wave
            pad_mask (BoolTensor): B x T_wave
            task_id (LongTensor): N >= 1
        """

        feat, pad_mask = self.forward_feature(wave, pad_mask) #! only feature extractor

        if self.task_emb_type not in ["none", "expand-last", "self-hidden", "layer-wise"]:
            if task_id is None:
                task_id = self.generate_task_id(feat.device)
            elif isinstance(task_id, list):
                task_id = torch.LongTensor(task_id).to(feat.device)
            task_embs = self.task_embedding(task_id)
            # N x D
            n_sz = len(task_id)
        else:
            n_sz = 1
        b_sz, t_sz, _ = feat.shape

        if self.task_emb_type == "add":
            # Add embs to feature
            if self.post_extract_proj is not None:
                feat_final = self.post_extract_proj(feat)
            else:
                feat_final = feat
            feat_final = feat_final.unsqueeze(1) + task_embs.unsqueeze(0).unsqueeze(2)
        elif self.task_emb_type == "concat":
            # Concatenates embs to feature
            feat_final = torch.cat(
                [
                    feat.unsqueeze(1).expand(-1, n_sz, -1, -1),
                    task_embs.unsqueeze(0).unsqueeze(2).expand(b_sz, -1, t_sz, -1),
                ],
                dim=-1,
            )
            if self.post_extract_proj is not None:
                feat_final = self.post_extract_proj(feat_final)
        else:
            if self.post_extract_proj is not None:
                feat_final = self.post_extract_proj(feat)
            else:
                feat_final = feat
            feat_final = feat_final.unsqueeze(1)
        # feat_final: B x N x T x D or B x 1 x T x D

        pad_mask = pad_mask.unsqueeze(1).expand(-1, n_sz, -1).reshape(b_sz * n_sz, t_sz)
        # BN x T
        feat_final = feat_final.reshape(b_sz * n_sz, t_sz, -1)
        # BN x T x D

        layer_hiddens = []
        if self.config.encoder_layers > 0:
            get_hidden_tmp = (
                True if (self.task_emb_type == "self-hidden") else get_hidden
            )
            hidden, layer_hiddens, attn_hiddens = self.encoder( #! encoder layer(only 2 layers)
                feat_final, ~pad_mask.bool(), get_hidden=True
            ) #! hidden [12,752,768]
        else:
            hidden = self.encoder(feat_final)

        if not no_pred and self.task_emb_type != "layer-wise":
            if self.task_emb_type == "self-hidden":
                pred = torch.stack([feat_final] + layer_hiddens, dim=1)
            else:
                pred = self.output_layer(hidden).reshape(b_sz, n_sz, t_sz, -1) #! output layer: Linear + GELU + SplitLinear
            # B x N x T x D
        else:
            pred = None

        if (not no_pred) and self.task_emb_type == "expand-last":
            assert n_sz == 1, n_sz
            pred = (
                pred.squeeze(1)
                .reshape(b_sz, t_sz, self.n_tasks, -1)
                .permute(0, 2, 1, 3)
            )
            # B x N x T x D

        if (not no_pred) and self.task_emb_type == "layer-wise": 
            pred = []
            for i, proj in enumerate(self.output_layer):
                result = proj(layer_hiddens[i])
                pred.append(result)
            pred = torch.stack(pred, dim=1)

        embeddings = self.get_embeddings(hidden) # B x T x E

        if get_hidden:
            return feat, feat_final, pred, pad_mask, layer_hiddens, embeddings
        
        else:
            return feat, feat_final, pred, pad_mask, embeddings

    def cal_pad_mask(self, pad_mask, max_len):
        """Calculates pad mask after conv."""
        pad_len = (pad_mask > 0).sum(1).long()
        for _, k_size, s_size in self.conv_layers:
            pad_len = torch.div((pad_len - k_size), s_size, rounding_mode="trunc") + 1

        new_pad_mask = torch.ones(
            (pad_mask.shape[0], max_len), dtype=pad_mask.dtype, device=pad_mask.device
        )

        for idx in range(pad_len.shape[0]):
            new_pad_mask[idx, pad_len[idx] :] = 0

        return new_pad_mask

    def generate_task_id(self, device):
        return torch.arange(self.n_tasks, device=device, dtype=torch.long)

    def get_embeddings(self, x):
        return self.final_proj(x)

