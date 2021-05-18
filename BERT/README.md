# This section is mainly used to train a BERT for evalution or MMD critic.

To install the environment for training BERT `bash requirements.sh`. 

To train a BERT

Here train/valid/test is the train folder that with .txt or .npy extension files inside.

```
python3 main.py \
    --overwrite_output_dir \
    --output_dir=output_dir \
    --train_dir=../data/maestro_magenta_s5_t3/train \
    --eval_dir=../data/maestro_magenta_s5_t3/valid \
    --test_dir=../data/maestro_magenta_s5_t3/test \
    --vocab_file=../data/maestro_magenta_s5_t3/vocab.txt \
    --num_hidden_layers=5 \
    --event_type=magenta \
    --model_type=bert \
    --block_size=20 \
    --tokenizer_name=midi_tokenizer \
    --do_train \
    --evaluate_during_training \
    --do_eval \
    --mlm
```

```
python3 -m torch.distributed.launch \
    --nproc_per_node=4 main.py \
     --overwrite_output_dir \
    --output_dir=output_dir \
    --train_dir=../data/maestro_magenta_s5_t3/train \
    --eval_dir=../data/maestro_magenta_s5_t3/valid \
    --test_dir=../data/maestro_magenta_s5_t3/test \
    --vocab_file=../data/maestro_magenta_s5_t3/vocab.txt \
    --num_hidden_layers=5 \
    --event_type=magenta \
    --model_type=bert \
    --block_size=20 \
    --tokenizer_name=midi_tokenizer \
    --do_train \
    --evaluate_during_training \
    --do_eval \
    --mlm
```

To kill torch.distributed.launch processes
```bash
kill $(ps aux | grep "main.py" | grep -v grep | awk '{print $2}')
```




