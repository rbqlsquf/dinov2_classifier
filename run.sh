#bin/bash

CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 \
/root/vlm_classification/dinov2/eval/linear.py \
  --output-dir /root/vlm_classification/out_classifier/out/dinov2_vitg14_reg4_pretrain \
  --batch-size 1024 \
  --num-workers 8 \
  --epochs 10 \
  --epoch-length 40 \
  --save-checkpoint-frequency 20 \