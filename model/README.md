# Run Experiments with the Maestro Dataset

We store the configurations inside `.yml` files. Here is an example: [experiment_baseline.yml](./experiment_baseline.yml).

```bash
python3 -m torch.distributed.launch --nproc_per_node=1 ./train.py \
    --data_dir ../data/maestro_magenta_s5_t3 \
    --cfg ./training_config/experiment_baseline.yml \
    --work_dir exp_dir
```

```bash
python3 -m torch.distributed.launch --nproc_per_node=1 ./train.py \
    --data_dir ../data/maestro_magenta_s5_t3 \
    --cfg ./training_config/experiment_spanbert.yml \
    --work_dir exp_dir
```


Change ```nproc_per_node = 4```, if you want to run on four GPUs.


To kill torch.distributed.launch processes
```bash
kill $(ps aux | grep "train.py" | grep -v grep | awk '{print $2}')
```


Generate text files from a trained model

```bash
python3 generate.py --model_dir exp_dir/20200518-232203 --out_dir generated_long/ --num 5 --gen_len 5000
```

Generate MIDIs from text files in the generated folder

```bash
python3 ../data/music_encoder.py --input_folder generated_long --output_folder generated_long_midi --mode to_midi

```