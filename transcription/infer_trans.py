import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mir_eval
from transcription.data import MAESTRO_V3
from pathlib import Path
import pickle
import mido
import torch as th
from types import SimpleNamespace
from torch.utils.data import DataLoader
from tqdm import tqdm
from transcription.pretrain import ARModel as ARModel2
from transcription.model_trans import TransModel


def schedule(t_max, a_min = 0.5, a_max=0.99):
    # incresing schedule from a_min to a_max
    alpha = a_min + (a_max - a_min) * (np.arange(t_max) / (t_max - 1))
    return alpha

def process_segment(feature, model, cond_init, steps):
    cond_frame = cond_init
    for iter in range(steps):
        mask = (th.rand(cond_frame.shape[0], cond_frame.shape[1], 88) < mask_schedule[iter]).cpu()
        cond = cond_frame * mask
        # with th.autocast(device_type='cuda', dtype=th.float16, enabled=True):
        frame_out  = model(feature.cuda(), cond.to(th.int).cuda(), mask.to(th.int).cuda())

        frame_cat = cond_frame * mask + frame_out.argmax(dim=-1).detach().cpu() * ~mask
        cond_frame = frame_cat
    return frame_cat

    

if __name__ == '__main__':
    steps = pow(2, 8)
    iters = [pow(2, e) for e in range(9)]
    mask_schedule = schedule(steps, 0.8, 1)
    seg_len = 800
    overlap = 50
    device = 'cuda'

    infer_path = 'runs/PAR_v2_230524-181231_PAR_v2_new2_herby/model_200k_0.9019_eval_MAESTRO_V3'
    trans_path = 'trans_eval/2'

    dataset = MAESTRO_V3(groups=['test'], seed=0, random_sample=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # load pretrain
    pretrain_model = ARModel2()
    ckp = th.load('model_130k_0.9140.pt')
    pretrain_model.load_state_dict(ckp['model_state_dict'], strict=False)
    pretrain_model.eval()
    pretrain_model.to(device)
    for param in pretrain_model.parameters():
        param.requires_grad = False
    
    # load transduction model
    ckp = th.load(Path('runs/NATTEN_240226-173823_w7xlaolu/model_170k_0.9106.pt'))
    config2 = {}
    for k, v in ckp.items():
        if k != 'model_state_dict':
            config2.update({k: v})
    config2 = SimpleNamespace(**config2)
    model2 = TransModel(config2)
    model2.load_state_dict(ckp['model_state_dict'])
    model2.to(device)

    for batch in tqdm(dataloader):
        save_path = Path(trans_path) / (Path(batch['path'][0]).stem + '.pt') 
        if save_path.exists():
            continue
        path = Path(infer_path) / (Path(batch['path'][0]).stem + '.npz')
        with np.load(path) as f:
            pred = th.from_numpy(f['pred']).argmax(-1)
        
        audio = batch['audio'].to(device)
        label = batch['label'][:,1:].to(device)
        vel = batch['velocity'][:,1:].to(device)
        n_step = label.shape[1]
        
        pred_holder = th.zeros_like(label)

        features = pretrain_model(audio)
        n_seg = (n_step - overlap) // (seg_len - overlap) + 1
        for seg in tqdm(range(n_seg)):
            start = seg * (seg_len - overlap)
            end = start + seg_len
            if seg == n_seg - 1:
                end = n_step
            feature = features[:, start:end]
            cond_init = pred[start:end].unsqueeze(0)
            seg_pred = process_segment(feature, model2, cond_init, steps)
            if seg == 0:
                pred[:end-overlap//2] = seg_pred[:, :-overlap//2]
            elif seg == n_seg - 1:
                pred[start+overlap//2:] = seg_pred[:, overlap//2:]
            else:
                pred[start+overlap//2:end-overlap//2] = seg_pred[:, overlap//2:-overlap//2]
        
        save_path = Path(trans_path) / (Path(batch['path'][0]).stem + '.pt') 
        th.save(pred, save_path)
        

        

