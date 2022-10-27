"""
convet a s3prl distiller model to fairseq model
"""

import sys
import torch
import s3prl.optimizers

ori_model_path = '/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/result/pretrain/distill_fithubert_fbank_20ms/fairseq-pretrain-without-encoder.pt'
new_model_path = '/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/result/pretrain/distill_fithubert_fbank_20ms/fairseq-pretrain.pt'

ori_model = torch.load(ori_model_path)

ori_keys = list(ori_model['model'].keys())
print(ori_keys)
print(type(ori_keys))

for idx in range(len(ori_keys)):
    if ori_keys[idx][0: len('encoder')] == 'encoder':
        ori_model['model']['feature_extractor.' + ori_keys[idx]] = ori_model['model'][ori_keys[idx]]
        print('add feature_extractor.' + ori_keys[idx])

torch.save(ori_model, new_model_path)
print(f'save model to {new_model_path}')
