lang=python

CUDA_VISIBLE_DEVICES=2,3 python run.py \
--output_dir ../$lang/model \
--model_type roberta \
--config_name microsoft/codebert-base \
--model_name_or_path microsoft/codebert-base \
--tokenizer_name roberta-base \
--do_train \
--train_data_file ../dataset/$lang/train.jsonl \
--eval_data_file ../dataset/$lang/valid.jsonl \
--epoch 2 \
--block_size 256 \
--train_batch_size 32 \
--eval_batch_size 64 \
--learning_rate 5e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 123456 \
2>&1| tee train.log