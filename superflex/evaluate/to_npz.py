import os
import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from superflex.superflex import SuperFlex
from superflex.utils.predictions_handler import PredictionHandler
from superflex.data.dataloader import ShapeNet, Scene, ABO, ASE_Object, denormalize_outdict, denormalize_points
from typing import Dict, Any
from tqdm import tqdm

def main(cfg: DictConfig) -> None:

    device = cfg.get('device', 'cuda')
    # Dataloader
    if cfg.dataset == 'shapenet':
        dataset = ShapeNet(split=cfg.dataloader.split, cfg=cfg)
        filename = f'{cfg.dataset}_{cfg.dataloader.split}.npz'
        z_up = False
    elif cfg.dataset == 'abo':
        dataset = ABO(split=cfg.dataloader.split, cfg=cfg)
        filename = f'{cfg.dataset}_{cfg.dataloader.split}.npz'
        z_up = False
    elif cfg.dataset == 'ase_object':
        dataset = ASE_Object(cfg.ase_object.split, cfg=cfg)
        filename = f'{cfg.dataset}_{cfg.ase_object.split}.npz'
        z_up = False
    elif cfg.dataset == 'scene':
        dataset = Scene(cfg=cfg)
        filename = f'{cfg.scene.name}.npz'
        z_up = cfg.scene.z_up

    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    dataloader = DataLoader(dataset, batch_size=cfg.dataloader.batch_size, shuffle=False, num_workers=cfg.dataloader.num_workers)
    ckp_path = os.path.join(cfg.checkpoints_folder, cfg.checkpoint_file)
    config_path = os.path.join(cfg.checkpoints_folder, 'config.yaml')
    if not os.path.isfile(ckp_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckp_path}")
    checkpoint = torch.load(ckp_path, map_location=device, weights_only=False)
    with open(config_path) as f:
        configs = OmegaConf.load(f)

    model = SuperFlex(configs.superflex).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        for i, b in tqdm(enumerate(dataloader)):
            points = b['points'].to(device).float()
            b['translation'] = b['translation'].to(device)
            b['scale'] = b['scale'].to(device)
            outdict = model(points)
            outdict = denormalize_outdict(outdict, b['translation'], b['scale'], z_up)
            points = denormalize_points(points, b['translation'], b['scale'], z_up)
            names = b.get('model_id', np.arange(points.shape[0]))
            if i == 0:
                pred_handler = PredictionHandler.from_outdict(outdict, points, names)
            else:
                pred_handler.append_outdict(outdict, points, names)

    pred_handler.save_npz(os.path.join(cfg.output_dir, filename)) # this step takes a lot of time (~1 minute)
    print(f"Results saved to {os.path.join(cfg.output_dir, filename)}")


if __name__ == "__main__":
    @hydra.main(version_base=None, config_path="../../configs/evaluate", config_name="save_npz")
    def run_main(cfg: DictConfig):
        main(cfg)
    run_main()