To do evaluation

Please download the pretrained checkpoints into the same machine and run the function and put the numpy files under folder of inference

```
python3 bert_score.py \
    --model_name_or_path=../BERT/checkpoint-1969000 \
    --vocab_file=../BERT/magenta_vocab_file.txt \
    --event_type=magenta

```

