"""
    Distiller_JOINT Model
    Author: Changli Tang (https://github.com/TCL606)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..wav2vec2.wav2vec2_model import (
    MultiheadAttention,
    SamePad,
    get_activation_fn,
    ConvFeatureExtractionModel,
    GradMultiply,
    LayerNorm,
    TransposeLast
)


def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


class SplitLinear(nn.Module):
    """Split Linear Layer"""

    def __init__(self, in_dim, in_split, out_dim):
        super().__init__()

        self.in_dim = in_dim  # Din
        self.in_split = in_split  # N
        self.out_dim = out_dim  # Dout

        if in_split > 1:
            # weight = torch.zeros((1, 1, self.in_split, self.in_dim, self.out_dim))
            weight = torch.zeros((self.in_split, self.in_dim, self.out_dim))
            self.weight = nn.Parameter(weight, requires_grad=True)
            nn.init.uniform_(self.weight, -(self.in_dim ** -0.5), self.in_dim ** -0.5)

            bias = torch.zeros((1, 1, self.in_split, self.out_dim))
            self.bias = nn.Parameter(bias, requires_grad=True)
            nn.init.uniform_(self.bias, -(self.in_dim ** -0.5), self.in_dim ** -0.5)
        else:
            self.layer = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, x: torch.Tensor): #! [12, 752, 1536]
        # x: shape = B x T x NDin

        if self.in_split == 1:
            return self.layer(x)
        else:
            x = x.reshape(x.shape[0], x.shape[1], self.in_split, 1, self.in_dim) #! [12, 752, 2, 1, 768]
            # x: B x T x N x 1 x Din

            out = torch.einsum("...klm,kmn->...kln", x, self.weight).squeeze(3)
            # out: B x T x N x Dout
            out = out + self.bias #! [12, 752, 2, 768]

            return out.reshape(x.shape[0], x.shape[1], -1)


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
        attention_type: str = "original",
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = get_activation_fn(activation_fn)
        self.attention_type = attention_type
        if attention_type == "original":
            self.self_attn = MultiheadAttention(
                self.embedding_dim,
                num_attention_heads,
                dropout=attention_dropout,
                self_attention=True,
            )
        elif attention_type == "sparse":
            from fairseq.modules.sparse_multihead_attention import (
                SparseMultiheadAttention,
            )

            self.self_attn = SparseMultiheadAttention(
                self.embedding_dim,
                num_attention_heads,
                dropout=attention_dropout,
                self_attention=True,
                stride=32,
                expressivity=16,
            )
        elif attention_type == "dynamic":
            from fairseq.modules import DynamicConv

            self.self_attn = DynamicConv(
                self.embedding_dim,
                kernel_size=31,
                padding_l=15,
                num_heads=num_attention_heads,
                weight_dropout=0.0,
                weight_softmax=True,
                bias=True,
            )
        else:
            raise NotImplementedError(f"Unknown attention type {attention_type}")

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward_self_attn(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = True,
    ):
        if self.attention_type in ["original", "sparse"]:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
                attn_mask=self_attn_mask,
            )
        elif self.attention_type == "dynamic":
            x = self.self_attn(x)
            attn = None

        return x, attn

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = True,
        att_args=None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.forward_self_attn(
                x,
                self_attn_mask=self_attn_mask,
                need_weights=True,
                self_attn_padding_mask=self_attn_padding_mask,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.forward_self_attn(
                x,
                self_attn_mask=self_attn_mask,
                need_weights=True,
                self_attn_padding_mask=self_attn_padding_mask,
            )

            x = self.dropout1(x)
            x = residual + x
            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, attn


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim

        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=args.conv_pos,
            padding=args.conv_pos // 2,
            groups=args.conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (args.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        if args.extractor_mode == 'default':
            self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
            self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())

        else:
            self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
            self.pos_conv = nn.Sequential(
                    *[nn.Sequential(
                        self.pos_conv, 
                        SamePad(args.conv_pos), 
                        nn.GELU(),
                        nn.Sequential(
                                TransposeLast(),
                                LayerNorm(self.embedding_dim, elementwise_affine=False),
                                TransposeLast()
                            )
                        ) for i in range (5)] 
                    )

        print(f"[TransformerEncoder] - Attention type = {args.attention_type}")
        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                    num_attention_heads=args.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    activation_fn=args.activation_fn,
                    layer_norm_first=args.layer_norm_first,
                    attention_type=args.attention_type,
                )
                for _ in range(args.encoder_layers)
            ]
        )

        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None, attn_mask=None, get_hidden=True):
        x, layer_results, attn_results = self.extract_features(
            x, padding_mask, attn_mask, get_hidden=get_hidden
        )

        if self.layer_norm_first:
            x = self.layer_norm(x)

        return x, layer_results, attn_results

    def extract_features(self, x, padding_mask=None, attn_mask=None, get_hidden=True):

        if padding_mask is not None:
            x[padding_mask] = 0

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        attn_results = []
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if 1:
                x, attn = layer(
                    x,
                    self_attn_padding_mask=padding_mask,
                    need_weights=True,
                    self_attn_mask=attn_mask,
                )
                if get_hidden:
                    attn_results.append(attn)
                    layer_results.append(x.transpose(0, 1))

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, layer_results, attn_results
