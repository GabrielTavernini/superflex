<p align="center">
 <h1 align="center">SuperFlex: Deformable Superquadrics for Point Cloud Decomposition</h2></h1>
<p align="center">
<a href="#">Gabriel Tavernini</a><sup>1*</sup>,
<a href="https://elisabettafedele.github.io/">Elisabetta Fedele</a><sup>1*</sup>,
<a href="https://sites.google.com/site/tiagonovellodebrito">Tiago Novello</a><sup>2,3</sup>,
<a href="https://geometry.stanford.edu/?member=guibas">Leonidas Guibas</a><sup>2</sup>,
<a href="https://people.inf.ethz.ch/pomarc/">Marc Pollefeys</a><sup>1</sup>,
<a href="https://francisengelmann.github.io/">Francis Engelmann</a><sup>4</sup>
<br>
<sup>1</sup>ETH Zurich,
<sup>2</sup>Stanford University,
<sup>3</sup>IMPA
<sup>4</sup>USI Lugano <br>
</p>
<h3 align="center"><a href="https://github.com/gabrieltavernini/superflex">Code</a> | <a href="#">Paper</a> | <a href="https://superflex3d.github.io">Project Page</a> </h3>
<div align="center"></div>
</p>
<p align="center"> 
<a href="">
<img src="https://superflex3d.github.io/static/figures/teaser.jpg" alt="Logo" width="100%">
</a>
</p>
<p align="center">
<strong>SuperFlex</strong> expands the expressive power and applicability of superquadric decompositions by utilizing tapering and bending.
</p>
<br>


## 🚀 Quick Start

### Environment Setup

Clone the repository and set up the environment:

```bash
git clone https://github.com/gabrieltavernini/superflex.git
cd superflex

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Build sampler (required for training only)
python setup_sampler.py build_ext --inplace
```

### Download Pre-trained Models

Download the checkpoints:

```bash
bash scripts/download_checkpoints.sh
```

Alternatively, you can download the individual folders using the links below.

| Dataset | Robust | Supervised | Link |
|:--------|:-----------:|:-----:|:-----|
| ShapeNet | ❌ | ❌ | [superflex](https://drive.google.com/drive/folders/1Ospu8mFz4kEQf5a6pnOOx87z-xWR6kQe?usp=sharing) |
| ShapeNet + ASE | ✅ | ❌ |[robust unsup.](https://drive.google.com/drive/folders/1bw-HHwQYKC62uUHF9t8QVdx_XVibESQb?usp=sharing) |
| ShapeNet + ASE | ✅ | ✅ |[robust](https://drive.google.com/drive/folders/1cHbtZLToQRontte6CyzvQpL5IHUqk-x4?usp=sharing) |

### Inference Example
Once downloaded the checkpoints you can run an inference example by doing:
```bash
python demo_viser.py
```

<p align="center">
  <img src="https://superflex3d.github.io/static/figures/viser/demo0.jpg" width="32%" />
  <img src="https://superflex3d.github.io/static/figures/viser/demo1.jpg" width="32%" />
  <img src="https://superflex3d.github.io/static/figures/viser/demo2.jpg" width="32%" />
</p>

### Optimization Example
You can run an optimization example by doing:
```bash
python demo_optimization.py
```

<p align="center">
  <img src="https://superflex3d.github.io/static/figures/viser/optimization_start.jpg" width="49%" />
  <img src="https://superflex3d.github.io/static/figures/viser/optimization_end.jpg" width="49%" />
</p>

## 🪑 Inference on ShapeNet 

### Download Data

Download the ShapeNet dataset (73.4 GB):

```bash
bash scripts/download_shapenet.sh
```

The dataset will be saved to `data/ShapeNet/`. After having downloaded ShapeNet and the checkpoints, the following project structure is expected:
```
superflex/
├── checkpoints/          # Checkpoints storage
│   ├── unsup_robust/     # Checkpoint and config for unsup_robust
│   ├── robust/           # Checkpoint and config for robust
│   └── superflex/        # Checkpoint and config for superflex
├── data/                 # Dataset storage
│   └── ShapeNet/         # ShapeNet dataset
├── examples/             # Inference example
│   └── chair.glb         # ABO chair mesh
│   └── lamp.ply          # ShapeNet lamp pointcloud
├── scripts/              # Utility scripts
├── superflex/            # Main package
├── train/                # Training scripts
└── requirements.txt      # Dependencies
```
### Inference and visualization on test set

Generate and visualize results on ShapeNet test set:
```bash
bash scripts/run_on_shapenet.sh 
```
> **Note:** Saving the .npz file and mesh generation may take time depending on the size of the dataset and of the chosen resolution for the superquadrics, respectively.

### Optimization

Optimize and visualize results on ShapeNet test set:
```bash
bash scripts/run_optimization.sh 
```

### Training (optional)

If you want to retrain the network yourself you can either opt for single or multi-gpu training as follows.

**Single GPU training:**
```bash
python train/train.py
```

**Multi-GPU training (4 GPUs):**
```bash
torchrun --nproc_per_node=4 train/train.py
```
> **Note:** Weights & Biases is disabled by default but you can activate it in the [training config](configs/train.yaml).



## 🏡 Inference on Full Scenes 
We assume you have the .ply files of all the segmented objects in a single folder OBJECTS_SCENE_DIR. Fill required fields in the [script](scripts/run_on_scene.sh), following the given instructions. Now you are ready to run inference by doing:
```bash
bash scripts/run_on_scene.sh 
```

## 🙏  Acknowledgements
We adapted some codes from some awesome repositories including [superquadric_parsing](https://github.com/paschalidoud/superquadric_parsing), [CuboidAbstractionViaSeg](https://github.com/SilenKZYoung/CuboidAbstractionViaSeg), [volumentations](https://github.com/kumuji/volumentations), [LION](https://github.com/nv-tlabs/LION), [occupancy_networks](https://github.com/autonomousvision/occupancy_networks), and [convolutional_occupancy_networks](https://github.com/autonomousvision/convolutional_occupancy_networks). Thanks for making codes and data public available.

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests. For more specific questions or collaborations, please contact [Gabriel](mailto:gtavernini@ethz.ch) and [Elisabetta](mailto:efedele@ethz.ch).
