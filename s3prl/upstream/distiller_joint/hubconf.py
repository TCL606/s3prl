"""
    Distiller_JOINT Model
    Author: Changli Tang (https://github.com/TCL606)
"""

import os

from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert


def distiller_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def distiller_url(ckpt, refresh=False, *args, **kwargs):
    """
    The model from url
        ckpt (str): URL
        refresh (bool): whether to download ckpt/config again if existed
    """
    return distiller_local(_urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def distilhubert(refresh=False, *args, **kwargs):
    """
    DistilHuBERT
    """
    return distilhubert_base(refresh=refresh, *args, **kwargs)


def distilhubert_base(refresh=False, *args, **kwargs):
    """
    DistilHuBERT Base
    Default model in https://arxiv.org/abs/2110.01900
    """
    kwargs[
        "ckpt"
    ] = "https://www.dropbox.com/s/hcfczqo5ao8tul3/disilhubert_ls960_4-8-12.ckpt?dl=1"
    return distiller_url(refresh=refresh, *args, **kwargs)

def distilhubert_half(*args, **kwargs):
    ckpt="/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/result/pretrain/distill_hubert/states-110000.ckpt"
    print("Use distilhubert_half model")
    return distiller_local(ckpt=ckpt)

def distilhubert_full(*args, **kwargs):
    ckpt="/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/result/pretrain/distill_hubert/states-epoch-8.ckpt"
    print("Use distilhubert_full model")
    return distiller_local(ckpt=ckpt)

def distilhubert_celoss(*args, **kwargs):
    ckpt="/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/result/pretrain/distill_hubert_w_celoss/states-epoch-1.ckpt"
    print("Use distilhubert_celoss model")
    return distiller_local(ckpt=ckpt)

def distilhubert_baseline(*args, **kwargs):
    ckpt="/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/result/pretrain/distill_hubert_fp16/states-epoch-1.ckpt"
    print("Use distilhubert_baseline model")
    return distiller_local(ckpt=ckpt)

def distilhubert_w_emb_loss_only(*args, **kwargs):
    ckpt="/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/result/pretrain/distill_hubert_w_embloss_only/states-epoch-1.ckpt"
    print("Use distilhubert_w_emb_loss_only model")
    return distiller_local(ckpt=ckpt)

def distill_hubert_w_embloss_high_temp(*args, **kwargs):
    ckpt="/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/result/pretrain/distill_hubert_w_embloss_high_temp/states-epoch-1.ckpt"
    print("Use distill_hubert_w_embloss_high_temp model")
    return distiller_local(ckpt=ckpt)

def distill_hubert_alpkd_l2l(*args, **kwargs):
    ckpt="/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/result/pretrain/distill_hubert_alpkd_l2l/states-epoch-1.ckpt"
    print("Use distill_hubert_alpkd_l2l model")
    return distiller_local(ckpt=ckpt)

def distill_hubert_type2_student_no_temp(*args, **kwargs):
    ckpt="/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/result/pretrain/distill_hubert_type2_student_no_temp/states-epoch-1.ckpt"
    print("Use distill_hubert_type2_student_no_temp model")
    return distiller_local(ckpt=ckpt)

def distill_hubert_type3_student_no_temp(*args, **kwargs):
    ckpt="/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/result/pretrain/distill_hubert_type3_student_no_temp/states-epoch-1.ckpt"
    print("Use distill_hubert_type3_student_no_temp model")
    return distiller_local(ckpt=ckpt)

def fithubert_baseline(*args, **kwargs):
    ckpt="/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/result/pretrain/distill_fithubert_fp16/states-epoch-1.ckpt"
    print("Use fithubert_baseline model")
    return distiller_local(ckpt=ckpt)

def fithubert_baseline_new(*args, **kwargs):
    ckpt="/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/result/pretrain/distill_fithubert_fp16_new/states-epoch-1.ckpt"
    print("Use fithubert_baseline_new model")
    return distiller_local(ckpt=ckpt)

def fithubert_new_type2(*args, **kwargs):
    ckpt="/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/result/pretrain/distill_fithubert_new_type2/states-epoch-1.ckpt"
    print("Use fithubert_new_type2")
    return distiller_local(ckpt=ckpt)

def fit_hubert_type3_student_no_temp(*args, **kwargs):
    ckpt="/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/result/pretrain/distill_fithubert_type3_student_no_temp/states-epoch-1.ckpt"
    print("Use distill_fithubert_type3_student_no_temp model")
    return distiller_local(ckpt=ckpt)

def distill_hubert_kmeans_layer12(*args, **kwargs):
    ckpt="/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/result/pretrain/distill_hubert_kmeans_layer12/states-epoch-1.ckpt"
    print("Use distill_hubert_kmeans_layer12 model")
    return distiller_local(ckpt=ckpt)

def distill_hubert_kmeans_layer9(*args, **kwargs):
    ckpt="/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/result/pretrain/distill_hubert_kmeans_layer9/states-epoch-1.ckpt"
    print("Use distill_hubert_kmeans_layer9 model")
    return distiller_local(ckpt=ckpt)

def distill_hubert_kmeans_layer6(*args, **kwargs):
    ckpt="/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/result/pretrain/distill_hubert_kmeans_layer6/states-epoch-1.ckpt"
    print("Use distill_hubert_kmeans_layer6 model")
    return distiller_local(ckpt=ckpt)

def distill_hubert_kmeans_layer3(*args, **kwargs):
    ckpt="/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/result/pretrain/distill_hubert_kmeans_layer3/states-epoch-1.ckpt"
    print("Use distill_hubert_kmeans_layer3 model")
    return distiller_local(ckpt=ckpt)

def distill_hubert_weight_0_4(*args, **kwargs):
    ckpt="/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/result/pretrain/distill_hubert_0.4/states-epoch-1.ckpt"
    print("Use distill_hubert_0.4 model")
    return distiller_local(ckpt=ckpt)

def distill_hubert_weight_0_2(*args, **kwargs):
    ckpt="/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/result/pretrain/distill_hubert_0.2/states-epoch-1.ckpt"
    print("Use distill_hubert_0.2 model")
    return distiller_local(ckpt=ckpt)

def distill_hubert_weight_0_2_type2(*args, **kwargs):
    ckpt="/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/result/pretrain/distill_hubert_0.2_type2/states-epoch-1.ckpt"
    print("Use distill_hubert_0.2 model")
    return distiller_local(ckpt=ckpt)

def distill_hubert_weight_0_4_type2(*args, **kwargs):
    ckpt="/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/result/pretrain/distill_hubert_0.4_type2/states-epoch-1.ckpt"
    print("Use distill_hubert_0.4 model")
    return distiller_local(ckpt=ckpt)

def distill_hubert_weight_0_6_type2(*args, **kwargs):
    ckpt="/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/result/pretrain/distill_hubert_0.6_type2/states-epoch-1.ckpt"
    print("Use distill_hubert_0.6 model")
    return distiller_local(ckpt=ckpt)

def distill_hubert_weight_0_8_type2(*args, **kwargs):
    ckpt="/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/result/pretrain/distill_hubert_0.8_type2/states-epoch-1.ckpt"
    print("Use distill_hubert_0.8 model")
    return distiller_local(ckpt=ckpt)
    
def distill_hubert_type2_no_tau(*args, **kwargs):
    ckpt="/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/result/pretrain/distill_hubert_type2_no_tau/states-epoch-1.ckpt"
    print("Use distill_hubert_type2_no_tau model")
    return distiller_local(ckpt=ckpt)