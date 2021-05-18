Symbolic Music Generation with Transformer-GANs

Code for the paper "[Symbolic Music Generation with Transformer-GANs](https://assets.amazon.science/36/e6/95f355a24df983dfcd2fe6b5ad2a/symbolic-music-generation-with-transformer-gans.pdf)" (AAAI 2021)

If you use this code, please cite the paper using the bibtex reference below.
```
@inproceedings{transformer-gan,
    title={Symbolic Music Generation with Transformer-GANs},
    author={Aashiq Muhamed and Liang Li and Xingjian Shi and Suri Yaddanapudi and Wayne Chi and Dylan Jackson and Rahul Suresh and Zachary C. Lipton and Alexander J. Smola},
    booktitle={35th AAAI Conference on Artificial Intelligence, {AAAI} 2021},
    year={2021},
}
```

## Requirements
- Python 3.6+
- Pytorch
- Transformers

You can install all required Python packages with `bash requirements.sh`. 

## Datasets, switching inside `data` folder

* Downloaded data

```bash
bash get_data.sh
```

* Run `music_encoder.py` to generate the encoded numpy files
  * Messages stating that pitches are out of range are expected behavior


```bash
python3 music_encoder.py --encode_official_maestro \  
    --mode midi_to_npy \  
    --pitch_transpose_lower -3 \  
    --pitch_transpose_upper 3 \  
    --output_folder ./maestro_magenta_s5_t3  
```

## Train and Generate: switching inside `model` folder

* Train a Transformer XL (No GAN)

```bash
python3 -m torch.distributed.launch --nproc_per_node=4 ./train.py \
    --data_dir ../data/maestro_magenta_s5_t3 \
    --cfg ./training_config/experiment_baseline.yml \
    --work_dir exp_dir
```

* Train a Transformer XL (with GAN)

```bash
python3 -m torch.distributed.launch --nproc_per_node=4 ./train.py \
    --data_dir ../data/maestro_magenta_s5_t3 \
    --cfg ./training_config/experiment_spanbert.yml \
    --work_dir exp_dir
```

* Generate unconditional samples 

```
# generate unconditional samples
python3 generate.py --inference_config inference_config/inference_unconditional.yml
```

Note, if you are loading an old config.yml file which includes None/" " inside, please change it to a string 'Null' to make sure you can do cfg.merge_from_file.

* Extend music to generate conditional samples

```
# generate conditional samples
python3 generate.py --inference_config inference_config/inference_conditional.yml

```

1. Please set condition_len as well as condition_file
2. Change memlen and genlen. memlen=genlen is recommended

## Post process for data (convert `.txt` to `.mid`)

* Run the following to get midi files from txt files
  * Use `--mode to_midi` for text file conversions. Use `--mode npy_to_midi` for numpy file conversions.

```bash
python3 ../data/music_encoder.py --input_folder ./Output_Uncondtitionl --output_folder ./Output_Uncondtitionl_MIDI --mode to_midi
python3 ../data/music_encoder.py --input_folder ./Output_Condtitionl --output_folder ./Output_Condtitionl_MIDI --mode to_midi
```

different methods inside music_encoder

* encoder.to_text(input.mid, output.txt)
* encoder.from_text(input.txt, out.mid)
* encoder.encode_vocab(input.mid) return list of ids
* encoder.decoder_vocab(list(ids)) return out.mid
* encoder.to_text_argumentaion(input.mid, output.txt)





