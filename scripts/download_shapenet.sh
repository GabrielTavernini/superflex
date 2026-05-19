#!/bin/bash

set -e

# Default download directory
DEFAULT_DIR="./data"

echo "Do you want to store the dataset in a custom directory? (y/n)"
read -r custom_dir

if [ "$custom_dir" = "y" ]; then
    echo "Enter full path to your desired directory (e.g., /media/username/data):"
    read -r YOUR_DIR

    mkdir -p "$YOUR_DIR"
    cd "$YOUR_DIR"
    echo "Downloading dataset to $YOUR_DIR..."
    wget https://s3.eu-central-1.amazonaws.com/avg-projects/occupancy_networks/data/dataset_small_v1.1.zip
    unzip dataset_small_v1.1.zip '*pointcloud.npz'
    unzip dataset_small_v1.1.zip '*points.npz'
    unzip dataset_small_v1.1.zip '*.lst'
    rm dataset_small_v1.1.zip

    # Link the ShapeNet directory to ./data
    cd -
    mkdir -p ./data
    ln -s "$YOUR_DIR"/ShapeNet ./data/ShapeNet
else
    mkdir -p "$DEFAULT_DIR"
    cd "$DEFAULT_DIR"
    echo "Downloading dataset to $DEFAULT_DIR..."
    wget https://s3.eu-central-1.amazonaws.com/avg-projects/occupancy_networks/data/dataset_small_v1.1.zip
    unzip dataset_small_v1.1.zip '*pointcloud.npz'
    unzip dataset_small_v1.1.zip '*points.npz'
    unzip dataset_small_v1.1.zip '*.lst'
    rm dataset_small_v1.1.zip
fi

echo "Done."