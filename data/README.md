# Create event representations from scratch

- Download the MAESTRO dataset

```bash
bash get_data.sh
```
- Run `music_encoder.py` to generate the encoded numpy files.


```bash
# Generate with the Magenta Event-based representation
python3 music_encoder.py --encode_official_maestro \
                        --mode midi_to_npy \
                        --pitch_transpose_lower -3 \
                        --pitch_transpose_upper 3 \
                        --output_folder ./maestro_magenta_s5_t3

```
