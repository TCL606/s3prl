'''
extract the linear projection from a distill-finetune model
'''

import torch

pretrain_model_path = '/mnt/lustre/sjtu/home/xc915/superb/upstream_model/distill_hubert_finetune_fairseq.pt'
linear_projection_path = '/mnt/lustre/sjtu/home/xc915/superb/upstream_model/linear_projection/hubert_finetune_distill_linear.pt'

# load all keys from w2v_encoder.w2v_model in finetuned model to pretrain model
pretrain_model = torch.load(pretrain_model_path)

class LinearProj(torch.nn.Module):
    #! now the size of linear projection is fixed: (768, 32)
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(768, 32)

    def forward(self, x):
        x = self.linear(x)
        return x

linear_proj = LinearProj()
print(linear_proj.state_dict().keys())

linear_proj.linear.weight = torch.nn.Parameter(pretrain_model['model']['linear_projection.weight'])
linear_proj.linear.bias = torch.nn.Parameter(pretrain_model['model']['linear_projection.bias'])

torch.save(linear_proj.state_dict(), linear_projection_path)
print(f'save model to {linear_projection_path}')