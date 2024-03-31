lang=python     #programming language
batch_size=64
beam_size=10
source_length=256
target_length=128
data_dir=../dataset
output_dir=../model/$lang
dev_file=$data_dir/$lang/valid.jsonl
test_file=$data_dir/$lang/test.jsonl
test_model=$output_dir/epoch_10/subject_model.pth #checkpoint for test

CUDA_VISIBLE_DEVICES=2,3 python run.py \
--do_test --model_type roberta \
--model_name_or_path microsoft/codebert-base \
--load_model_path $test_model \
--dev_filename $dev_file \
--test_filename $test_file \
--output_dir $output_dir \
--max_source_length $source_length \
--max_target_length $target_length \
--beam_size $beam_size \
--eval_batch_size $batch_size