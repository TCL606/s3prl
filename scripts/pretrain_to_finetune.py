'''
load weight from a pretrained model to a finetuned model
'''

from collections import OrderedDict
import torch

finetune_model_path = '/mnt/lustre/sjtu/home/xc915/superb/upstream_model/hubert-finetune/hubert-base-finetune-100h.pt'
pretrain_model_path = '/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/result/pretrain/distill_fithubert_w_all_loss/fairseq-pretrain.pt'
new_model_path = '/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/result/pretrain/distill_fithubert_w_all_loss/fairseq_finetune.pt'

# load all keys from w2v_encoder.w2v_model in finetuned model to pretrain model

finetune_model = torch.load(finetune_model_path)
pretrain_model = torch.load(pretrain_model_path)

str_length = len('w2v_encoder.w2v_model.')

# del finetune_model['model']
# finetune_model['model'] = OrderedDict()

# for key in pretrain_model['model'].keys():
#     finetune_model['model']['w2v_encoder.w2v_model.'+ key] = pretrain_model['model'][key]
#     print(f'[maintain] convert {key} to w2v_encoder.w2v_model.{key}')

# finetune_model = torch.load(finetune_model_path)
# pretrain_model = torch.load(pretrain_model_path)

# str_length = len('w2v_encoder.w2v_model.')

for key in pretrain_model['model'].keys():
    if 'w2v_encoder.w2v_model.' + key in finetune_model['model'].keys():
        finetune_model['model']['w2v_encoder.w2v_model.'+ key] = pretrain_model['model'][key]
        print(f'[maintain] convert {key} to w2v_encoder.w2v_model.{key}')
    else:
        print(f'[ignore] ignore {key}')

# final linear proj
finetune_model['model']['w2v_encoder.proj.weight'] = pretrain_model['model']['linear_projection.weight']
print(f'[maintain] convert linear_projection.weight to w2v_encoder.proj.weight')
finetune_model['model']['w2v_encoder.proj.bias'] = pretrain_model['model']['linear_projection.bias']
print(f'[maintain] convert linear_projection.bias to w2v_encoder.proj.bias')

torch.save(finetune_model, new_model_path)
print(f'save new model to {new_model_path}')