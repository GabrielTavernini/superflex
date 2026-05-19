import os
import torch
import torch.distributed as dist
import hydra
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer
from utils import build_model, build_optimizer, build_scheduler, build_dataloaders, build_loss, set_seed, setup_ddp, is_main_process
try:
    import wandb
except ImportError:
    wandb = None

def to_str_dict(d):
    if isinstance(d, dict):
        return {str(k): to_str_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [to_str_dict(i) for i in d]
    else:
        return d

@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    is_distributed = int(os.environ.get('WORLD_SIZE', 1)) > 1
    if is_distributed:
        local_rank = setup_ddp()
        device = torch.device(f'cuda:{local_rank}')
    else:
        local_rank = 0
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    set_seed(cfg.seed + local_rank)

    run = None
    if cfg.use_wandb and wandb is not None and is_main_process():
        if hasattr(cfg.checkpoints, 'run_id'):
            run = wandb.init(
                project=cfg.wandb.project,
                name=cfg.run_name,
                id=cfg.checkpoints.run_id,
                resume='must'
            )
        else:
            run = wandb.init(
                project=cfg.wandb.project,
                name=cfg.run_name
            )

    model = build_model(cfg).to(device)
    
    dataloaders, train_sampler = build_dataloaders(cfg, is_distributed=is_distributed)

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, len(dataloaders['train'])) # None if disabled
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    checkpoint_path = getattr(cfg.checkpoints, 'resume_from', None)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        if cfg.checkpoints.keep_epoch:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"Resumed from checkpoint {checkpoint_path} at epoch {start_epoch}, best_val_loss={best_val_loss}")
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_fn = build_loss(cfg).to(device)

    

    cfg.trainer.save_path = os.path.join(cfg.trainer.save_path, cfg.run_name)
    if is_main_process() and not os.path.exists(cfg.trainer.save_path):
        os.makedirs(cfg.trainer.save_path, exist_ok=True)
    # save cfg in save path as yaml
    if is_main_process():
        with open(os.path.join(cfg.trainer.save_path, "config.yaml"), "w") as fp:
            OmegaConf.save(cfg, f=fp)
        if run is not None:
            artifact = wandb.Artifact(name="config", type="config")
            artifact.add_file(os.path.join(cfg.trainer.save_path, "config.yaml"))
            run.log_artifact(artifact)

    trainer = Trainer(model, optimizer, scheduler, dataloaders, loss_fn, cfg.trainer, run, start_epoch=start_epoch, best_val_loss=best_val_loss, is_distributed=is_distributed, train_sampler=train_sampler)
    trainer.train()
    if run is not None:
        run.finish()
    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()