CKPT_FOLDER=$1
if [ -z "$CKPT_FOLDER" ]; then
  echo "Usage: $0 <ckpt_folder>"
  exit 1
fi
DATASET=shapenet

python -m superflex.evaluate.to_npz \
  "dataset=$DATASET" \
  "checkpoints_folder=checkpoints/$CKPT_FOLDER" \
  "output_dir=data/output_npz/$CKPT_FOLDER"

python -m superflex.evaluate.evaluate \
  "dataloader.dataset=$DATASET" \
  "+source_folder=$CKPT_FOLDER"