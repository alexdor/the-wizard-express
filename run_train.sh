#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="5,6,7"
export RAYON_RS_NUM_CPUS="15"

echo $CACHE_DIR

poetry run python run_qa.py \
  --model_name_or_path distilbert-base-cased \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 5 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./tmp/debug_squad/ \
  --cache_dir $CACHE_DIR \
  --preprocessing_num_workers 15
