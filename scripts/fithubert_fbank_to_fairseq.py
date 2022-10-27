"""
convet a s3prl distiller model to fairseq model
"""

import sys
import torch
import s3prl.optimizers

distill_model_path = '/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/result/pretrain/distill_fitfbank_2stage_only_kldiv/states-epoch-1.ckpt'
original_model_path = '/mnt/lustre/sjtu/home/xc915/superb/upstream_model/hubert_base.ls960.pt'
new_model_path = '/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/result/pretrain/distill_fitfbank_2stage_only_kldiv/fairseq-pretrain.pt'
sys.modules["optimizers"] = s3prl.optimizers
distill_model = torch.load(distill_model_path)

print('distill model')
for key in distill_model['Distiller'].keys():
    print(key, distill_model['Distiller'][key].shape)
original_model = torch.load(original_model_path) 

print('hubert pretrain model')
for key in original_model['model']:
    if(hasattr(original_model['model'][key], 'shape')):
        print(key, original_model['model'][key].shape)
original_model['model'] = distill_model['Distiller']

# for hubert
original_model['args'].encoder_embed_dim = 480
original_model['args'].encoder_ffn_embed_dim = 480
original_model['args'].arch = 'hubert_fbank'

if 'final_proj.weight' in original_model['model']:
    del original_model['model']['final_proj.weight']
    del original_model['model']['final_proj.bias']

# for data2vec
# original_model['cfg']['model']['encoder_embed_dim'] = 480
# original_model['cfg']['model']['encoder_ffn_embed_dim'] = 480

torch.save(original_model, new_model_path)
print(f'save model to {new_model_path}')
