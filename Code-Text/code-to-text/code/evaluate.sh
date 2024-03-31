lang=python
gold_file=../model/$lang/dev.gold
output_file=../model/$lang/dev.output

python evaluator.py \
$gold_file < $output_file