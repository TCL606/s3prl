'''
load weight from a pretrained model to a finetuned model
'''

import torch

finetune_model_path = '/mnt/lustre/sjtu/home/xc915/superb/upstream_model/hubert-finetune/hubert_0_1_true.pt'
pretrain_model_path = '/mnt/lustre/sjtu/home/xc915/superb/upstream_model/distill_hubert_finetune_fairseq.pt'
new_model_path = '/mnt/lustre/sjtu/home/xc915/superb/upstream_model/pretrain_to_finetune/hubert_finetune_distill_random_linear.pt'

# load all keys from w2v_encoder.w2v_model in finetuned model to pretrain model

finetune_model = torch.load(finetune_model_path)
pretrain_model = torch.load(pretrain_model_path)

str_length = len('w2v_encoder.w2v_model.')

for key in pretrain_model['model'].keys():
    if 'w2v_encoder.w2v_model.' + key in finetune_model['model'].keys():
        finetune_model['model']['w2v_encoder.w2v_model.'+ key] = pretrain_model['model'][key]
        print(f'[maintain] convert {key} to w2v_encoder.w2v_model.{key}')
    else:
        print(f'[ignore] ignore {key}')

# final linear proj
finetune_model['model']['w2v_encoder.proj.weight'] = torch.nn.Parameter(torch.randn((32, 768)))
print(f'[init] randomly initialized w2v_encoder.proj.weight')
finetune_model['model']['w2v_encoder.proj.bias'] = torch.nn.Parameter(torch.randn((32)))
print(f'[init] randomly initialized w2v_encoder.proj.bias')

torch.save(finetune_model, new_model_path)
print(f'save new model to {new_model_path}')