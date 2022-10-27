"""
convet a s3prl distiller model to fairseq model
"""

import sys
import torch
import s3prl.optimizers



distill_model_path = '/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/result/pretrain/distill_hubert_sort_only_rec/states-epoch-1.ckpt'
original_model_path = '/mnt/lustre/sjtu/home/xc915/superb/upstream_model/hubert_base.ls960.pt'
new_model_path = '/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/result/pretrain/distill_hubert_sort_only_rec/fairseq-pretrain.pt'
sys.modules["optimizers"] = s3prl.optimizers
distill_model = torch.load(distill_model_path)

print('distill model')
for key in distill_model['Distiller'].keys():
    print(key, distill_model['Distiller'][key].shape)
original_model = torch.load(original_model_path) 

print('hubert/data2vec pretrain model')
for key in original_model['model']:
    if(hasattr(original_model['model'][key], 'shape')):
        print(key, original_model['model'][key].shape)
print(original_model['model'].keys())
print(distill_model['Distiller'].keys())
original_model['model'] = distill_model['Distiller']

# for hubert
original_model['args'].encoder_layers = 2
original_model['args'].arch = 'hubert_fbank'
del original_model['model']['final_proj.weight']
del original_model['model']['final_proj.bias']

# for data2vec
# original_model['cfg']['model']['encoder_layers'] = 2

torch.save(original_model, new_model_path)
print(f'save model to {new_model_path}')

