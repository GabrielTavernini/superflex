mkdir -p checkpoints/superflex
mkdir -p checkpoints/robust
mkdir -p checkpoints/unsup_robust

gdown --folder 1Ospu8mFz4kEQf5a6pnOOx87z-xWR6kQe -O checkpoints/superflex
gdown --folder 1cHbtZLToQRontte6CyzvQpL5IHUqk-x4 -O checkpoints/robust
gdown --folder 1bw-HHwQYKC62uUHF9t8QVdx_XVibESQb -O checkpoints/unsup_robust

# for training we also need the superdec checkpoint
mkdir -p checkpoints/superdec
gdown 1Nsgtm_nCyp6qbRgnenoJVqL88eS1GXmC -O checkpoints/superdec/config.yaml
gdown 1ypCViehSOzkCFL6dcCDfdPRzuj_MIayz -O checkpoints/superdec/ckpt.pt
