import torch
import torch.distributed as dist
import os
from torch.utils.data import DataLoader
from superflex.superflex import SuperFlex
from superflex.data.dataloader import ShapeNet, ABO, ASE_Object
from superflex.loss.loss import Loss
from torch.optim import Adam
import random
import numpy as np
import hydra
from torch.utils.data.distributed import DistributedSampler

def setup_ddp():
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return local_rank

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def build_model(cfg):
    # keep compatibility with superdec configs
    model_cfg = cfg.superflex if 'superflex' in cfg else cfg.superdec
    model = SuperFlex(model_cfg)
    return model.cuda() if torch.cuda.is_available() else model

def build_optimizer(cfg, model):
    if cfg.optimizer.only_heads:
        return Adam(model.heads.parameters(), lr=cfg.optimizer.lr, betas=cfg.optimizer.betas, weight_decay=cfg.optimizer.weight_decay)
    elif cfg.optimizer.all_but_heads:
        heads_params = set(model.heads.parameters())
        params = [p for p in model.parameters() if p not in heads_params]
        return Adam(params, lr=cfg.optimizer.lr, betas=cfg.optimizer.betas, weight_decay=cfg.optimizer.weight_decay)
    else:
        return Adam(model.parameters(), lr=cfg.optimizer.lr, betas=cfg.optimizer.betas, weight_decay=cfg.optimizer.weight_decay)

def build_scheduler(cfg, optimizer, step_per_epoch):
    if not cfg.optimizer.enable_scheduler:
        return None
    cfg.scheduler.steps_per_epoch = step_per_epoch
    return hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)

def build_dataloaders(cfg, is_distributed=False):
    if cfg.dataset == 'shapenet':
        train_ds = ShapeNet(split='train', cfg=cfg)
        val_ds = ShapeNet(split='val', cfg=cfg)
    elif cfg.dataset == 'abo':
        train_ds = ABO(split='train', cfg=cfg)
        val_ds = ABO(split='val', cfg=cfg)
    elif cfg.dataset == 'ase_object':
        train_ds = ASE_Object(split='train', cfg=cfg)
        val_ds = ASE_Object(split='val', cfg=cfg)
    else:
        raise ValueError(f"Unsupported dataset {cfg.dataset}")

    if is_distributed:
        train_sampler = DistributedSampler(train_ds)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds, batch_size=cfg.trainer.batch_size, shuffle=shuffle,
        num_workers=cfg.trainer.num_workers, pin_memory=True, sampler=train_sampler
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.trainer.batch_size, shuffle=False,
        num_workers=cfg.trainer.num_workers, pin_memory=True
    )
    return {'train': train_loader, 'val': val_loader}, train_sampler if is_distributed else None


def build_loss(cfg):
    return Loss(cfg.loss)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
