import heapq
import argparse
import os
import sys
import tempfile
from pathlib import Path
import json
import csv
from types import SimpleNamespace
from collections import defaultdict
from tqdm import tqdm

import torch as th
import torch.distributed as dist
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import StepLR
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import wandb

from adabelief_pytorch import AdaBelief

from .model import ARModel
from .data import MAESTRO_V3
from .loss import FocalLoss
from .evaluate import evaluate
from .utils import summary

th.autograd.set_detect_anomaly(True)


def remove_progress(captured_out):
    lines = (line for line in captured_out.splitlines() if ('it/s]' not in line) and ('s/it]' not in line))
    return '\n'.join(lines)

default_dict = dict(
    n_mels=704,
    n_fft=4096,
    f_min=27.5,
    f_max=8000,
    cnn_unit=48,
    lstm_unit=48,
    hidden_per_pitch=48,
    fc_unit=768,
    batch_size=12,
    win_fw=4,
    win_bw=0,
    model='PAR',
    dataset='MAESTRO_V3',
    seq_len=160256,
    n_workers=4,
    lr=1e-3,
    n_epoch = 100,
    random_condition=True,
    valid_interval=10000,
    debug=False,
    seed=1000,
    resume_id=None,
    iteration=250000
    
    )
   
   
def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group(backend='nccl',
                        init_method='tcp://127.0.0.1:23456',
                        world_size=world_size,
                        rank=rank)

def cleanup():
    dist.destroy_process_group()
    
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 256),
            nn.ReLU(),
            nn.Linear(256, 5)
        )
    def forward(self, x):
        return self.net(x)
    
def get_dataset(config, split, random_sample, transform):
    if config.dataset == 'MAESTRO_V3':
        return MAESTRO_V3(groups=split, sequence_length=config.seq_len, 
                          random_sample=random_sample, transform=transform)

class ModelSaver():
    def __init__(self, logdir, config, order='lower', n_keep=3, resume=False):
        self.logdir = Path(logdir)
        self.order = order
        assert order in ['lower', 'higher']
        self.config = config
        self.n_keep = n_keep
        self.top_n = []
        self.best_ckp = None
        self.last_ckp = None
        self.last_opt = None
        self.last_step = -1

        if resume:
            with open(self.logdir / 'checkpoint.csv', "r") as f:
                reader = csv.reader(f, delimiter=',')
                self.top_n = [(el[0], float(el[1])) for el in list(reader)]
            self.best_ckp = self.top_n[0][0]
            lastest = np.argmax([int(el[0].split('_')[1]) for el in self.top_n])
            self.last_ckp = self.top_n[lastest][0]
            self.last_step = int(self.last_ckp.split('_')[0])
            self.last_opt = self.save_name_opt(self.last_step)

    def save_model(self, model, save_name, ddp):
        save_dict = self.config.__dict__
        state_dict = model.module.state_dict() if ddp else model.state_dict()
        save_dict['model_state_dict'] = state_dict
        th.save(save_dict, self.logdir / save_name)
        self.last_ckp = save_name

    def update_optim(self, optimizer, step):
        opt_name = self.save_name_opt(step)
        th.save(optimizer.state_dict(), self.logdir / opt_name)
        last_opt = self.logdir / self.save_name_opt(self.last_step)
        if last_opt.exists():
            last_opt.unlink()
        self.last_opt = opt_name

    def save_name_opt(self, step):
        if step > 1000:
            return f'opt_{step//1000}k.pt'
        else:
            return f'opt_{step}.pt'
    
    def save_name(self, step, score):
        if step > 1000:
            return f'model_{step//1000}k_{score:.4f}.pt'
        else:
            return f'model_{step}_{score:.4f}.pt'
    
    def write_csv(self):
        with open(self.logdir / 'checkpoint.csv', "w") as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows([(el[0], el[1]) for el in self.top_n])
    
    def update(self, model, optimizer, step, score, ddp):
        save_name = self.save_name(step, score)
        self.save_model(model, save_name, ddp)
        self.update_optim(optimizer, step)
        self.top_n.append((save_name, score))
        self.update_top_n()
        self.last_step = step

    def update_top_n(self): 
        if len(self.top_n) <= self.n_keep:
            return
        if self.order == 'lower':
            reverse = False
        elif self.order == 'higher':
            reverse = True
        self.top_n.sort(key=lambda x: x[1], reverse=reverse)
        lowest = self.top_n[-1]
        if lowest[0] != self.last_ckp:
            (self.logdir / lowest[0]).unlink()
            self.top_n = self.top_n[:-1]
        self.best_ckp = self.top_n[0][0]


class Losses(nn.Module):
    def __init__(self):
        super().__init__()
        self.frame_loss_fn = FocalLoss(alpha=1.0, gamma=2.0) # In: B x C x *
        self.vel_loss_fn = nn.MSELoss(reduction='none') # In: B x *

    def forward(self, logit, vel, label, vel_label):
        frame_loss = self.frame_loss_fn(logit.permute(0, 3, 1, 2), label)
        onset_mask = ((label == 2) + (label == 4))>0
        vel_loss = self.vel_loss_fn(vel*onset_mask, th.true_divide(vel_label, 128)*onset_mask)

        return frame_loss, vel_loss
    

def train_step(model, batch, loss_fn, optimizer, scheduler, device, rank, step, config, run):
    for param in model.parameters():
        param.grad = None
    audio = batch['audio'].to(device)
    shifted_label = batch['label'].to(device)
    shifted_vel = batch['velocity'].to(device)
    last_onset_time = batch['last_onset_time'].to(device)
    last_onset_vel = batch['last_onset_vel'].to(device)
    frame_out, vel_out = model(audio, shifted_label[:, :-1], 
                                last_onset_time[:, :-1], last_onset_vel[:, :-1], 
                                random_condition=config.random_condition)
    # frame out: B x T x 88 x 5
    loss, vel_loss = loss_fn(frame_out, vel_out, shifted_label[:, 1:], shifted_vel[:, 1:])
    total_loss = loss.mean() + vel_loss.mean()
    total_loss.mean().backward()
    for parameter in model.parameters():
        clip_grad_norm_([parameter], 3.0)

    optimizer.step()
    scheduler.step()
    if rank == 0:
        run.log({"train": dict(frame_loss=loss.mean(), vel_loss=vel_loss.mean())}, step=step)
    
def valid_step(model, batch, loss_fn, device, config):
    audio = batch['audio'].to(device)
    shifted_label = batch['label'].to(device)
    shifted_vel = batch['velocity'].to(device)
    last_onset_time = batch['last_onset_time'].to(device)
    last_onset_vel = batch['last_onset_vel'].to(device)
    frame_out, vel_out = model(audio, shifted_label[:, :-1], 
                                last_onset_time[:, :-1], last_onset_vel[:, :-1], 
                                random_condition=False)
    # frame out: B x T x 88 x C
    loss, vel_loss = loss_fn(frame_out, vel_out, shifted_label[:, 1:], shifted_vel[:, 1:])
    validation_metric = defaultdict(list)
    for n in range(audio.shape[0]):
        sample = frame_out[n].argmax(dim=-1)
        metrics = evaluate(sample, shifted_label[n][1:], vel_out[n], shifted_vel[n][1:])
        for k, v in metrics.items():
            validation_metric[k].append(v)
    validation_metric['frame_loss'] = loss.mean(dim=(1,2))
    validation_metric['vel_loss'] = vel_loss.mean(dim=(1,2))
    
    return validation_metric

def transcribe(model, batch, loss_fn, device, config):
    audio = batch['audio'].to(device)
    shifted_label = batch['label'].to(device)
    shifted_vel = batch['velocity'].to(device)
    last_onset_time = batch['last_onset_time'].to(device)
    last_onset_vel = batch['last_onset_vel'].to(device)
    frame_out, vel_out = model(audio, shifted_label[:, :-1], 
                                last_onset_time[:, :-1], last_onset_vel[:, :-1], 
                                random_condition=False)
    
    return frame_out, vel_out

def train(rank, world_size, run, config, ddp=True):
    th.cuda.set_device(rank)
    if ddp:
        setup(rank, world_size)
    else:
        assert world_size == 1 and rank == 0
    device = f'cuda:{rank}'
    seed = config.seed + rank
    th.manual_seed(seed)
    np.random.seed(seed)

    model = ARModel(config).to(rank)
    if rank == 0:
        summary(model)
    # model = ToyModel().to(rank)
    if ddp:
        model = th.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank])
    optimizer = AdaBelief(model.parameters(), lr=config.lr, 
                          eps=1e-16, betas=(0.9,0.999), weight_decouple=True, 
                          rectify = False)

    if rank == 0:
        run.watch(model, log_freq=100)
    save_dir = run.dir
    if config.resume_id:
        model_saver = ModelSaver(save_dir, config, resume=True)
        ckp = th.load(model_saver.last_ckp)
        opt = th.load(model_saver.last_ckp)
        model.load_state_dict(
            th.load(ckp['model_state_dict'], map_location={'cuda:0': f'cuda:{rank}'})) 
        optimizer.load_state_dict(opt)
        step = model_saver.last_step
    else: 
        model_saver = ModelSaver(save_dir, config)
        step = 0
        
    scheduler = StepLR(optimizer, step_size=5000, gamma=0.95)
    train_set = get_dataset(config, ['train'], random_sample=True, transform=True)
    valid_set = get_dataset(config, ['validation'], random_sample=False, transform=False)
    if ddp:
        train_sampler = DistributedSampler(dataset=train_set, num_replicas=world_size, 
                                        rank=rank, shuffle=True)
        valid_sampler = DistributedSampler(dataset=valid_set, num_replicas=world_size, 
                                       shuffle=False)
    else:
        train_sampler=None
        valid_sampler=None
    data_loader_train = DataLoader(
        train_set, sampler=train_sampler,
        batch_size=config.batch_size//world_size,
        num_workers=config.n_workers,
        pin_memory=False,
        drop_last=True,
    )
    data_loader_valid = DataLoader(
        valid_set, sampler=valid_sampler,
        batch_size=config.batch_size//world_size,
        num_workers=config.n_workers,
        pin_memory=False,
        drop_last=False,
    )

    loss_fn = Losses()

    if rank == 0: loop = tqdm(range(step, config.iteration), total=config.iteration, initial=step)
    for epoch in range(10000):
        if ddp:
            data_loader_train.sampler.set_epoch(epoch)
        for batch in data_loader_train:
            if rank ==0: loop.update(1)
            step += 1
            model.train()
            train_step(model, batch, loss_fn, optimizer, scheduler, device, rank, step, config, run)

            if step % config.valid_interval == 0 or step == 5000:
                model.eval()

                validation_metric = defaultdict(list)
                with th.no_grad():
                    for batch in data_loader_valid:
                        batch_metric = valid_step(model, batch, loss_fn, device, config)
                        for k, v in batch_metric.items():
                            validation_metric[k].extend(v)
                valid_mean = defaultdict(list)
                if ddp:
                    output = [None for _ in range(world_size)]
                    dist.gather_object(validation_metric, output if rank==0 else None, dst=0)
                    if rank == 0:
                        for k,v in validation_metric.items():
                            if 'loss' in k:
                                valid_mean[k] = th.mean(th.cat([th.stack(el[k]).cpu() for el in output]))
                            else:
                                valid_mean[k] = np.mean(np.concatenate([el[k] for el in output]))
                else:
                    for k,v in validation_metric.items():
                        if 'loss' in k:
                            valid_mean[k] = th.mean(th.cat(th.stack(v).cpu()))
                        else:
                            valid_mean[k] = np.mean(np.concatenate(v))
                
                if rank == 0:
                    print('validation metric')
                    run.log({'valid':valid_mean}, step=step)
                    for key, value in valid_mean.items():
                        if key[-2:] == 'f1' or 'loss' in key or key[-3:] == 'err':
                            print(f'{key} : {value}')
                    model_saver.update(model, optimizer, step, valid_mean['frame_loss'], ddp=ddp)
                if ddp:
                    dist.barrier()
            if step >= config.iteration:
                wandb.finish()
                if ddp:
                    cleanup()
        
    
    
def run_demo(demo_fn, world_size, run, config):
    mp.spawn(demo_fn,
             args=(world_size, run, config),
             nprocs=world_size,
             join=True)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path)
    parser.add_argument('--model', type=str, default='PAR')
    parser.add_argument('--dataset', type=str, default='MAESTRO_V3')
    parser.add_argument('-b', '--batch_size', type=int, default=12)
    parser.add_argument('-c', '--cnn_unit', type=int, default=48)
    parser.add_argument('-l', '--lstm_unit', type=int, default=48)
    parser.add_argument('-p', '--hidden_per_pitch', type=int, default=48)
    parser.add_argument('-t', '--tag', type=str)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('--resume_id', type=str)
    parser.add_argument('--ddp', action='store_true')
    parser.add_argument('--no-ddp', dest='ddp', action='store_false')
    parser.set_defaults(ddp=True)
    
    args = parser.parse_args()
    config = default_dict
    if args.config:
        update_config = json.loads(args.config)
        config.update(update_config)
    config.update(vars(args))
    config = SimpleNamespace(**config)
    if config.debug:
        config.valid_interval=10
        config.iteration=100
    print(config)

    dataset = get_dataset(config, ['train', 'validation', 'test'], random_sample=False, transform=False)
    dataset.initialize()

    if args.resume_id:
        run = wandb.init('transcription', id=args.resume_id, resume="must")
    else:   
        run = wandb.init('transcription', config=config, tags=args.tag)
    
    if args.ddp:    
        n_gpus = th.cuda.device_count()
        assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
        world_size = n_gpus
        run_demo(train, world_size, run, config)
    else:
        train(rank=0, world_size=1, run=run, config=config, ddp=False)