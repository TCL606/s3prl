"""
    Distiller_Fbank Model
    Author: Changli Tang (https://github.com/TCL606)
"""

import os

from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert


def distiller_fbank_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)

def distill_fitfbank_2stage_w_kl(*args, **kwargs):
    ckpt = '/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/result/pretrain/distill_fitfbank_2stage_w_kl/states-epoch-1.ckpt'
    return distiller_fbank_local(ckpt=ckpt)
