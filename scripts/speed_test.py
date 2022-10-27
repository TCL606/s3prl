import s3prl.hub as hub
import s3prl
from s3prl.downstream.asr.dataset import SequenceDataset
from s3prl.downstream.asr.dictionary import Dictionary
from torch.utils.data import DataLoader
import torch
import time
from tqdm import tqdm

device = 'cpu'  
# model_fbank = getattr(hub, 'distill_fitfbank_2stage_w_kl')()  
# model_fbank = model_fbank.to(device)

# model_cnn = getattr(hub, 'distill_fithubert_new_type2')() 
# model_cnn = model_cnn.to(device)

model_sw = getattr(hub, 'distill_hubert_type2_no_tau')()
model_sw = model_sw.to(device)
threads = 4
torch.set_num_threads(threads)
validset = 'test-clean'

dictionary = Dictionary.load("/mnt/lustre/sjtu/home/xc915/superb/s3prl/s3prl/downstream/asr/char.dict")
dataset = SequenceDataset(
    validset, 
    32, 
    dictionary, 
    '/mnt/lustre/sjtu/home/xc915/superb/dataset/librispeech/LibriSpeech',
    '/mnt/lustre/sjtu/home/xc915/superb/s3prl/s3prl/data/len_for_bucket/',
    **{validset: [validset]}
)
dataloader = DataLoader(
            dataset, batch_size=1,
            shuffle=False, num_workers=12,
            collate_fn=dataset.collate_fn
        )
evaluate_ratio = float(1)
evaluate_steps = round(len(dataloader) * evaluate_ratio)

cnn_frontend = 0
cnn_transformer = 0
cnn_total = 0
fbank_frontend = 0
fbank_transformer = 0
fbank_total = 0
wavnum = 0

for batch_id, (wavs, *others) in enumerate(tqdm(dataloader)):
    wavs = [torch.FloatTensor(wav).to(device) for wav in wavs]
    wavnum += len(wavs)
    # with torch.no_grad():
    #     time_dict = model_cnn(wavs, get_time=True)
    #     cnn_frontend += time_dict['frontend_time']
    #     cnn_transformer += time_dict['transformer_time']
    #     cnn_total += time_dict['total_time']
    #     print(
    #         "cnn    frontend_time: ", time_dict['frontend_time'], 
    #         " transformer_time: ", time_dict['transformer_time'], 
    #         " total_time: ", time_dict['total_time']
    #     )
    # with torch.no_grad():
    #     time_dict = model_fbank(wavs, get_time=True)
    #     fbank_frontend += time_dict['frontend_time']
    #     fbank_transformer += time_dict['transformer_time']
    #     fbank_total += time_dict['total_time']
    #     print(
    #         "fbank  frontend_time: ", time_dict['frontend_time'], 
    #         " transformer_time: ", time_dict['transformer_time'], 
    #         " total_time: ", time_dict['total_time']
    #     )
        
    with torch.no_grad():
        time_dict = model_sw(wavs, get_time=True)
        fbank_frontend += time_dict['frontend_time']
        fbank_transformer += time_dict['transformer_time']
        fbank_total += time_dict['total_time']
        print(
            "S&W  frontend_time: ", time_dict['frontend_time'], 
            " transformer_time: ", time_dict['transformer_time'], 
            " total_time: ", time_dict['total_time']
        )
       
path = f'/mnt/lustre/sjtu/home/xc915/superb/ASR-scripts/testclean_sw.txt'
with open(path, 'w') as fp:
    fp.writelines(
        f"wave num: {wavnum} \n"
    )
    # fp.writelines(
    #     '\n'
    # )
    # fp.writelines(
    #     f"cnn \n frontend_time: {cnn_frontend} \n transformer_time: {cnn_transformer} \n total_time: {cnn_total} \n"
    # )
    # fp.writelines(
    #     '\n'
    # )
    # fp.writelines(
    #     f"fbank \n frontend_time: {fbank_frontend} \n transformer_time: {fbank_transformer} \n total_time: {fbank_total}"
    # )
    fp.writelines(
        '\n'
    )
    fp.writelines(
        f"S&W \n frontend_time: {fbank_frontend} \n transformer_time: {fbank_transformer} \n total_time: {fbank_total}"
    )
    


    