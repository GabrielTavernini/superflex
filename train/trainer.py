import os 
import numpy as np
from tqdm import tqdm
import torch
from utils import is_main_process
from superoptim.evaluation import compute_ious_sdf_from_outdict

try:
    import wandb
except ImportError:
    wandb = None
from wandb_viser import WandbViser

class Trainer:
    def __init__(self, model, optimizer, scheduler, dataloaders, loss_fn, ctx, wandb_run=None, start_epoch=0, best_val_loss=float('inf'), is_distributed=False, train_sampler=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloaders = dataloaders
        self.loss_fn = loss_fn
        self.ctx = ctx
        self.num_epochs = ctx.num_epochs
        self.save_path = ctx.save_path
        self.wandb_run = wandb_run
        self.wandb_viser = WandbViser(self.wandb_run)
        self.start_epoch = start_epoch
        self.is_distributed = is_distributed
        self.train_sampler = train_sampler


    def save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint and log to wandb."""
        if not is_main_process() or self.save_path is None: #TODO check whether this is ever called (I think it is not)
            return
        # Save model.module.state_dict() if using DDP
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'epoch': epoch,
            'val_loss': val_loss
        }
        ckpt_filename = f'epoch_{epoch+1}.pt'
        ckpt_path = os.path.join(self.save_path, ckpt_filename)
        torch.save(checkpoint, ckpt_path)
        if self.wandb_run is not None and wandb is not None:
            artifact = wandb.Artifact(ckpt_filename, type='model')
            artifact.add_file(ckpt_path)
            self.wandb_run.log_artifact(artifact)

    @torch.no_grad()
    def evaluate(self, epoch):
        """Evaluate model on validation set."""
        if self.is_distributed and torch.distributed.get_rank() != 0:
            return {}  # skip on non-zero ranks #TODO check whether this is ever called (I think it is not)
        self.model.eval()
        loader = self.dataloaders['val']
        pbar = tqdm(loader, desc=f"Eval  {epoch+1}/{self.num_epochs}", leave=False)

        total_loss = 0.0
        total_batches = 0
        avg_loss_dict = {}
        all_outputs = {
            'names': [], 'pc': [], 'assign_matrix': [], 'scale': [], 'rotation': [],
            'translation': [], 'exponents': [], 'exist': []
        }

        for batch in pbar:
            outdict = self.model(batch['points'].cuda().float())
            loss, loss_dict = self.loss_fn(batch, outdict)

            if 'points_iou' in batch and 'occupancies' in batch:
                points_iou = batch['points_iou'].cuda().float()
                occupancies = batch['occupancies'].cuda().bool()
                ious = compute_ious_sdf_from_outdict(outdict, points_iou, occupancies, device='cuda')
                loss_dict['iou'] = ious.mean().item()

            if total_batches % 10 == 0 and self.wandb_run is not None and wandb is not None:
                self.wandb_viser.accumulate_wandb_objects(epoch, outdict, batch, num_samples=2)

            total_loss += loss.item()
            total_batches += 1
            
            # Accumulate loss components
            for k, v in loss_dict.items():
                avg_loss_dict[k] = avg_loss_dict.get(k, 0.0) + v

            pbar.set_postfix({k: f"{v / total_batches:.4f}" for k, v in avg_loss_dict.items()})

        # Log accumulated objects
        if self.wandb_run is not None and wandb is not None:
            self.wandb_viser.log_accumulated_wandb_objects(epoch)

        # Compute averages
        for k in avg_loss_dict:
            avg_loss_dict[k] /= total_batches
        return avg_loss_dict

    def train_one_epoch(self, epoch):
        """Train for one epoch."""

        self.model.train()
        loader = self.dataloaders['train']
        if self.is_distributed and self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.num_epochs}", leave=False)

        total_loss = 0.0
        total_batches = 0
        avg_loss_dict = {}

        for batch in pbar:
            outdict = self.model(batch['points'].cuda().float())
            loss, loss_dict = self.loss_fn(batch, outdict)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            # Accumulate metrics
            total_loss += loss.item()
            total_batches += 1
            for k, v in loss_dict.items():
                avg_loss_dict[k] = avg_loss_dict.get(k, 0.0) + v

            pbar.set_postfix({k: f"{v:.4f}" for k, v in loss_dict.items()})

        # Compute averages
        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
        for k in avg_loss_dict:
            avg_loss_dict[k] /= total_batches

        # Log training metrics to wandb
        if self.wandb_run is not None and is_main_process():
            log_dict = {"train/loss": avg_loss}
            log_dict.update({f"train/{k}": v for k, v in avg_loss_dict.items()})
            
            # Log learning rate
            if self.optimizer.param_groups:
                lr = self.optimizer.param_groups[0].get('lr', None)
                if lr is not None:
                    log_dict["train/lr"] = lr
                    
            self.wandb_run.log(log_dict, step=epoch)

    def train(self):
        """Main training loop."""
        save_every = getattr(self.ctx, 'save_every_n_epochs', 1)
        evaluate_every = getattr(self.ctx, 'evaluate_every_n_epochs', 1)
        val_loss = None
        
        for epoch in range(self.start_epoch, self.num_epochs):
            # Training phase
            self.train_one_epoch(epoch)
            
            # Evaluation phase
            if is_main_process():
                do_eval = ((epoch + 1) % evaluate_every == 0) or (epoch == self.num_epochs - 1)
                if do_eval:
                    val_metrics = self.evaluate(epoch)
                    val_loss_val = val_metrics.get('loss', None) or list(val_metrics.values())
                    val_loss = val_loss_val[0] if val_loss_val else 0.0

                    # Log validation metrics to wandb
                    if self.wandb_run is not None:
                        self.wandb_run.log({f"val/{k}": v for k, v in val_metrics.items()}, step=epoch)

                do_save = ((epoch + 1) % save_every == 0) or (epoch == self.num_epochs - 1)
                if do_save: 
                    self.save_checkpoint(epoch, val_loss)