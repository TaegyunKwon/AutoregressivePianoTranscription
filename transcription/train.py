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
from datetime import datetime
import time

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
from .data import MAESTRO_V3, MAESTRO
from .loss import FocalLoss
from .evaluate import evaluate
from .utils import summary, CustomSampler
from .transcribe import PadCollate, transcribe

th.autograd.set_detect_anomaly(True)
os.environ["WANDB_DISABLE_SERVICE"] = "true"

def remove_progress(captured_out):
    lines = (line for line in captured_out.splitlines() if ('it/s]' not in line) and ('s/it]' not in line))
    return '\n'.join(lines)

default_dict = dict(
    n_mels=700,
    n_fft=4096,
    f_min=27.5,
    f_max=8000,
    cnn_unit=48,
    lstm_unit=48,
    hidden_per_pitch=48,
    fc_unit=768,
    shrink_channels=[4,1],
    batch_size=12,
    pitchwise_lstm=True,
    film=True,
    win_fw=4,
    win_bw=0,
    model='PAR',
    dataset='MAESTRO_V3',
    seq_len=160256,
    n_workers=4,
    lr=1e-3,
    n_epoch = 100,
    noisy_condition=True,
    valid_interval=10000,
    valid_seq_len=716800,
    debug=False,
    seed=1000,
    resume_dir=None,
    iteration=250000,
    port=23456
    
    )
   
   
def setup(rank, world_size, port=23456):
    # initialize the process group
    dist.init_process_group(backend='nccl',
                        init_method=f'tcp://127.0.0.1:{port}',
                        world_size=world_size,
                        rank=rank)

def cleanup():
    dist.destroy_process_group()
    
def get_dataset(config, split, sample_len=160256, random_sample=False, transform=False, load_mode='lazy'):
    if config.dataset == 'MAESTRO_V3':
        return MAESTRO_V3(groups=split, sequence_length=sample_len, 
                          random_sample=random_sample, transform=transform, load_mode=load_mode)
    elif config.dataset == 'MAESTRO_V1':
        return MAESTRO(groups=split, sequence_length=sample_len, 
                          random_sample=random_sample, transform=transform, load_mode=load_mode)
    else:
        raise KeyError

class ModelSaver():
    def __init__(self, config, order='lower', n_keep=3, resume=False):
        self.logdir = Path(config.logdir)
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
                self.top_n = [(el[0], float(el[1]), int(el[2])) for el in list(reader)]
            self.best_ckp = self.top_n[0][0]
            lastest = np.argmax([el[2] for el in self.top_n])
            self.last_ckp = self.top_n[lastest][0]
            self.last_step = self.top_n[lastest][2]
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
            writer.writerows([(el[0], el[1], el[2]) for el in self.top_n])
    
    def update(self, model, optimizer, step, score, ddp):
        save_name = self.save_name(step, score)
        self.save_model(model, save_name, ddp)
        self.update_optim(optimizer, step)
        self.top_n.append((save_name, score, step))
        self.update_top_n()
        self.write_csv()
        self.last_step = step

    def update_top_n(self): 
        if self.order == 'lower':
            reverse = False
        elif self.order == 'higher':
            reverse = True
        self.top_n.sort(key=lambda x: x[1], reverse=reverse)
        self.best_ckp = self.top_n[0][0]
        if len(self.top_n) > self.n_keep:
            del_list = self.top_n[self.n_keep:]
            self.top_n = self.top_n[:self.n_keep]
            for save_name, score, step in del_list:
                if self.last_ckp == save_name:
                    self.top_n.append((save_name, score, step))
                    continue
                (self.logdir / save_name).unlink()

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
    

def train_step(model, batch, loss_fn, optimizer, scheduler, device, config):
    for param in model.parameters():
        param.grad = None
    audio = batch['audio'].to(device)
    shifted_label = batch['label'].to(device)
    shifted_vel = batch['velocity'].to(device)
    last_onset_time = batch['last_onset_time'].to(device)
    last_onset_vel = batch['last_onset_vel'].to(device)
    frame_out, vel_out = model(audio, shifted_label[:, :-1], 
                                last_onset_time[:, :-1], last_onset_vel[:, :-1], 
                                random_condition=config.noisy_condition)
    # frame out: B x T x 88 x 5
    loss, vel_loss = loss_fn(frame_out, vel_out, shifted_label[:, 1:], shifted_vel[:, 1:])
    total_loss = loss.mean() + vel_loss.mean()
    optimizer.zero_grad()
    total_loss.mean().backward()
    for parameter in model.parameters():
        clip_grad_norm_([parameter], 3.0)

    optimizer.step()
    scheduler.step()
    return loss, vel_loss
    
def valid_step(model, batch, loss_fn, device, config):
    audio = batch['audio'].to(device)
    shifted_label = batch['label'].to(device)
    shifted_vel = batch['velocity'].to(device)
    # 'gt' sampling gives very poor results since some onsets are ignored, as the model doesn't
    # have to make it.
    frame_out, vel_out = model(audio, last_states=None, random_condition=False, sampling='argmax')
    # frame out: B x T x 88 x C
    loss, vel_loss = loss_fn(frame_out, vel_out, shifted_label[:, 1:], shifted_vel[:, 1:])
    validation_metric = defaultdict(list)
    for n in range(audio.shape[0]):
        sample = frame_out[n].argmax(dim=-1)
        metrics = evaluate(sample, shifted_label[n][1:], vel_out[n], shifted_vel[n][1:], band_eval=True)
        for k, v in metrics.items():
            validation_metric[k].append(v)
    validation_metric['frame_loss'] = loss.mean(dim=(1,2))
    validation_metric['vel_loss'] = vel_loss.mean(dim=(1,2))
    
    return validation_metric, frame_out, vel_out

def test_step(model, batch, device):
    audio = batch['audio'].to(device)
    # 'gt' sampling gives very poor results since some onsets are ignored, as the model doesn't
    # have to make it.
    frame_out, vel_out = model(audio, last_states=None, random_condition=False, sampling='argmax')
    # frame out: B x T x 88 x C
    frame_outs = [] 
    vel_outs = []
    test_metric = defaultdict(list)
    for n in range(audio.shape[0]):
        step_len = batch['step_len'][n]
        frame = frame_out[n][:step_len].detach().cpu()
        vel = vel_out[n][:step_len].detach().cpu()
        frame_outs.append(frame)
        vel_outs.append(vel)
        sample = frame.argmax(dim=-1)
        metrics = evaluate(sample, batch['label'][n].detach().cpu()[1:],
                           vel, batch['velocity'][n].detach().cpu()[1:], band_eval=False)
        for k, v in metrics.items():
            test_metric[k].append(v)
    
    return test_metric, frame_outs, vel_outs


def train(rank, world_size, config, ddp=True):
    th.cuda.set_device(rank)
    if ddp:
        setup(rank, world_size, port=config.port)
    else:
        assert world_size == 1 and rank == 0
    device = f'cuda:{rank}'
    seed = config.seed + rank
    th.manual_seed(seed)
    np.random.seed(seed)

    model = ARModel(config).to(rank)
    if config.resume_dir:
        model_saver = ModelSaver(config, resume=True, order='higher')
        step = model_saver.last_step
    else:  
        model_saver = ModelSaver(config, order='higher')
        step = 0
    if rank == 0:
        if config.resume_dir:
            run = wandb.init('transcription', id=config.id, resume="must", dir=config.logdir)
        else:   
            run = wandb.init('transcription', config=config, id=config.id, name=config.name, dir=config.logdir)
        summary(model)
    if ddp:
        model = th.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank])
    optimizer = AdaBelief(model.parameters(), lr=config.lr, 
                          eps=1e-16, betas=(0.9,0.999), weight_decouple=True, 
                          rectify = False, print_change_log=False)
    if config.resume_dir:
        ckp = th.load(model_saver.logdir / model_saver.last_ckp, map_location={'cuda:0':f'cuda:{rank}'})
        if ddp:
            model.module.load_state_dict(ckp['model_state_dict'])
        else:
            model.load_state_dict(ckp['model_state_dict'])
        del ckp
        if not config.eval:
            ckp_opt = th.load(model_saver.logdir / model_saver.last_opt)
            optimizer.load_state_dict(ckp_opt)
            del ckp_opt
        if ddp:
            dist.barrier()
        

    if rank == 0:
        run.watch(model, log_freq=1000)

    scheduler = StepLR(optimizer, step_size=5000, gamma=0.95)
    train_set = get_dataset(config, ['train'], sample_len=config.seq_len, 
                            random_sample=True, transform=config.noisy_condition, load_mode='lazy')
    valid_set = get_dataset(config, ['validation'], sample_len=config.valid_seq_len,
                            random_sample=False, transform=False, load_mode='lazy')
    if ddp:
        train_sampler = DistributedSampler(dataset=train_set, num_replicas=world_size, 
                                        rank=rank, shuffle=True)
        segments = np.split(np.arange(len(valid_set)),
                            np.arange(len(valid_set), step=config.batch_size//world_size))[1:]  # the first segment is []
        target_segments = [el for n, el in enumerate(segments) if n%world_size == rank]
        valid_sampler = CustomSampler(target_segments)
        data_loader_valid = DataLoader(
            valid_set, batch_sampler=valid_sampler,
            num_workers=config.n_workers,
            pin_memory=False,
        )
    else:
        train_sampler=None
        data_loader_valid = DataLoader(
            valid_set, sampler=None,
            batch_size=config.batch_size,
            num_workers=config.n_workers,
            pin_memory=False,
        )
    data_loader_train = DataLoader(
        train_set, sampler=train_sampler,
        batch_size=config.batch_size//world_size,
        num_workers=config.n_workers,
        pin_memory=False,
        drop_last=True,
    )

    loss_fn = Losses()

    if rank == 0: loop = tqdm(range(step, config.iteration), total=config.iteration, initial=step)
    for epoch in range(10000):
        if config.eval:
            break
        if ddp:
            data_loader_train.sampler.set_epoch(epoch)
        for batch in data_loader_train:
            step += 1
            if step > config.iteration:
                break
            if rank ==0: loop.update(1)
            model.train()
            loss, vel_loss= train_step(model, batch, loss_fn, optimizer, scheduler, device, config)
            if rank == 0:
                run.log({"train": dict(frame_loss=loss.mean(), vel_loss=vel_loss.mean())}, step=step)
            del loss, vel_loss, batch
            if step % config.valid_interval == 0 or step == 5000:
                model.eval()

                validation_metric = defaultdict(list)
                with th.no_grad():
                    for n_valid, batch in enumerate(data_loader_valid):
                        batch_metric, _, _ = valid_step(model, batch, loss_fn, device, config)
                        for k, v in batch_metric.items():
                            validation_metric[k].extend(v)
                        # for first batch, log image of feature map. shape(feature) = B C F L
                        if n_valid == 0 and rank == 0:
                            '''
                            # TODO: do this with hook
                            visual_range = 1000
                            for n in range(config.batch_size//world_size):
                                fig, axes = plt.subplots(config.cnn_unit//6, 6, figsize=(8, 10))
                                plt.axis('off')
                                for m in range(config.cnn_unit):
                                    axes[m//6, m%6].imshow(feature[n,m].numpy()[:,:visual_range], aspect='auto', origin='lower')
                                plt.subplots_adjust(wspace=0, hspace=0)
                                run.log({'valid':{f'fmap_{n}': plt}}, step=step)
                                plt.close()
                            '''

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
                            valid_mean[k] = th.mean(th.stack(v).cpu())
                        else:
                            valid_mean[k] = np.mean(np.concatenate(v))
                
                if rank == 0:
                    print(f'validation metric: step:{step}')
                    run.log({'valid':valid_mean}, step=step)
                    for key, value in valid_mean.items():
                        if key[-2:] == 'f1' or 'loss' in key or key[-3:] == 'err':
                            print(f'{key} : {value}')
                    valid_mean['metric/note/f1']
                    model_saver.update(model, optimizer, step, valid_mean['metric/note-with-offsets/f1'], ddp=ddp)
                if ddp:
                    dist.barrier()
        if step > config.iteration:
            break

    # Test phase
    model.eval()
    model_saver = ModelSaver(config, resume=True, order='higher')  # to load best model for all ranks
    SAVE_PATH = config.logdir / (Path(model_saver.best_ckp).stem + f'_eval_{config.dataset}')
    SAVE_PATH.mkdir(exist_ok=True)
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    ckp = th.load(model_saver.logdir / model_saver.best_ckp, map_location=map_location)
    if ddp:
        model.module.load_state_dict(ckp['model_state_dict'])
    else:
        model.load_state_dict(ckp['model_state_dict'])
    
    test_set = get_dataset(config, ['test'], sample_len=None,
                            random_sample=False, transform=False)
    test_set.sort_by_length()
    batch_size = 6 # 6 for PAR model, 12G RAM (8 blocked by 8G shm size)
    if ddp:
        segments = np.split(np.arange(len(test_set)),
                            np.arange(len(test_set), step=batch_size))[1:]  # the first segment is []
        target_segments = [el for n, el in enumerate(segments) if n%world_size == rank]
        test_sampler = iter(target_segments)
    else:
        test_sampler = None
    if config.debug:
        test_sampler = iter([[0,1]])
    data_loader_test = DataLoader(
        test_set, batch_sampler=test_sampler,
        num_workers=config.n_workers,
        pin_memory=False,
        collate_fn=PadCollate()
        )
    test_metrics = defaultdict(list)

    iterator = data_loader_test
    with th.no_grad():
        for batch in iterator:
            batch_metric, preds, vel_preds = test_step(model, batch, device)
            for k, v in batch_metric.items():
                test_metrics[k].extend(v)
            for n in range(len(preds)):
                pred = preds[n].detach().cpu().numpy()
                vel = vel_preds[n].detach().cpu().numpy()
                np.savez(Path(SAVE_PATH) / (Path(batch['path'][n]).stem + '.npz'), pred=pred, vel=vel)

    test_mean = defaultdict(list)
    if ddp:
        output = [None for _ in range(world_size)]
        dist.gather_object(test_metrics, output if rank==0 else None, dst=0)
        if rank == 0:
            for k,v in test_metrics.items():
                if 'loss' in k:
                    test_mean[k] = th.cat([th.stack(el[k]).cpu() for el in output])
                else:
                    test_mean[k] = np.concatenate([el[k] for el in output])
    else:
        for k,v in test_metrics.items():
            if 'loss' in k:
                test_mean[k] = th.cat(th.stack(v).cpu())
            else:
                test_mean[k] = np.concatenate(v)
        
    if rank == 0:
        with open((Path(SAVE_PATH) / f'summary_{config.dataset}.txt'), 'w') as f:
            string, count = summary(model) 
            f.write(string + '\n')
            print('test metric')
            run.log({'test':test_mean}, step=step)
            for key, value in test_mean.items():
                if 'loss' not in key:
                    _, category, name = key.split('/')
                    multiplier = 100
                    if 'err' in key:
                        multiplier=1
                    metric_string = f'{category:>32} {name:26}: {np.mean(value)*multiplier:.3f} +- {np.std(value)*multiplier:.3f}'
                    print(metric_string)
                    f.write(metric_string + '\n')
                else:
                    metric_string = f'{key:>32}: {th.mean(value)*100:.3f} +- {th.std(value)*100:.3f}'
                    print(metric_string)
                    f.write(metric_string + '\n')
        wandb.finish()
    if ddp:
        dist.barrier()
        cleanup()
        
    
    
def run_demo(demo_fn, world_size, config):
    mp.spawn(demo_fn,
            args=(world_size, config),
            nprocs=world_size,
            join=True)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path)
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('-b', '--batch_size', type=int)
    parser.add_argument('-c', '--cnn_unit', type=int)
    parser.add_argument('-l', '--lstm_unit', type=int)
    parser.add_argument('-p', '--hidden_per_pitch', type=int)
    parser.add_argument('-n', '--name', type=str)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--resume_dir', type=Path)
    parser.add_argument('--resume_id', type=str)
    parser.add_argument('--ddp', action='store_true')
    parser.add_argument('--no-ddp', dest='ddp', action='store_false')
    parser.add_argument('--port', type=int)
    parser.set_defaults(ddp=True)
    parser.set_defaults(eval=False)
    
    args = parser.parse_args()
    config = default_dict
    if args.config:
        with open(args.config, 'r') as j:
            update_config = json.load(j)
        print(update_config)
        config.update(update_config)
    for k, v in vars(args).items():
        if v is not None:
            config.update({k:v})
    config = SimpleNamespace(**config)
    if config.debug:
        config.valid_interval=5
        # config.valid_seq_len=160256
        config.iteration=50

    if args.resume_dir:
        id = args.resume_id
        config.id = id
        print(f'resume:{id}')
        config.logdir = args.resume_dir
    else:
        id = wandb.util.generate_id()
        config.id = id
        print(f'init:{id}')
        if hasattr(config, 'name'):
            config.logdir = Path('runs') / \
            ('_'.join([config.model, datetime.now().strftime('%y%m%d-%H%M%S'), config.name]))
        else:
            config.name=id
            config.logdir = Path('runs') / \
            ('_'.join([config.model, datetime.now().strftime('%y%m%d-%H%M%S'), id]))
        Path(config.logdir).mkdir(exist_ok=True)
    print(config)

    dataset = get_dataset(config, ['train', 'validation', 'test'], random_sample=False, transform=False)
    dataset.initialize()

    
    if args.ddp:    
        n_gpus = th.cuda.device_count()
        assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
        world_size = n_gpus
        run_demo(train, world_size, config)
    else:
        train(rank=0, world_size=1, config=config, ddp=False)