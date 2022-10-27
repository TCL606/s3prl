"""
delete transformer layers of a pretrained model, only maintain the layer 0 and 1
"""

import torch

old_model_path = '/mnt/lustre/sjtu/home/xc915/superb/upstream_model/hubert_base.ls960.pt'
new_model_path = '/mnt/lustre/sjtu/home/xc915/superb/upstream_model/hubert_0_1.pt'

old_model = torch.load(old_model_path) 
key_list = list(old_model["model"].keys())

for key in key_list:
    if('encoder.layers' in key):
        if('encoder.layers.0.' in key or 'encoder.layers.1.' in key):
            pass
        else:
            del old_model["model"][key]

for key in old_model["model"].keys():
    print(key)


old_model['args'].encoder_layers= 2

torch.save(old_model, new_model_path)
print(f'save model to {new_model_path}')