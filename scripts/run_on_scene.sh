#!/bin/bash
set -euo pipefail  # stop on error, undefined variable, or pipeline failure

# trap to print the failing command
trap 'echo "Error occurred at command: $BASH_COMMAND"' ERR

OBJECTS_SCENE_DIR=data/3f1e1610de/objects # path to the folder containing the .ply files of all the segmented objects in the scene
OUTPUT_NPZ_DIR=data/output_npz # path to the folder where to save the output .npz files
SCENE_NAME=3f1e1610de # name of the scene (used to name the output .npz file)
Z_UP=true

python superflex/utils/ply_to_npz.py --input_path="$OBJECTS_SCENE_DIR" --scene_name="$SCENE_NAME"

python superflex/evaluate/to_npz.py checkpoints_folder="checkpoints/unsup_robust" output_dir="$OUTPUT_NPZ_DIR" dataset=scene scene.name="$SCENE_NAME" scene.z_up="$Z_UP"

python superflex/visualization/object_visualizer.py dataset=scene split="$SCENE_NAME" npz_folder="$OUTPUT_NPZ_DIR"