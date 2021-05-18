# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
  
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import subprocess
from generate import main, parse_args
from utils.config_inference import get_default_cfg_inference
from shlex import quote

def generate_files(inference_cfg, prefix_files, sampling_technique, sampling_threshold, model_name, temperature,
                   time_extension=True, number_of_files=3):
    if time_extension:
        for prefix_file in prefix_files:
            inference_cfg.defrost()
            inference_cfg.INPUT.conditional_input_melody = prefix_file
            output_dir = os.path.join(os.path.join(survey_sample, 'new_samples_{}'.format(str(sampling_technique))),
                                      model_name + '_' + prefix_file.split('/')[-1][:-4])
            inference_cfg.INPUT.num_empty_tokens_to_ignore = 0
            inference_cfg.OUTPUT.output_txt_directory = output_dir
            inference_cfg.INPUT.num_midi_files = number_of_files
            inference_cfg.INPUT.time_extension = time_extension
            inference_cfg.SAMPLING.technique = sampling_technique
            inference_cfg.SAMPLING.threshold = sampling_threshold
            inference_cfg.INPUT.num_conditional_tokens = conditioned_len
            inference_cfg.SAMPLING.temperature = temperature
            inference_cfg.freeze()
            main(inference_cfg)
            cmd = "python3 ../data/music_encoder.py --input_folder {} --output_folder {} --mode " \
                  "to_midi".format(output_dir, output_dir + '_MIDI')
            returned_value = subprocess.call(quote(cmd), shell=True)  # returns the exit code in unix
    else:
        output_dir = os.path.join(os.path.join(survey_sample, 'new_samples_{}'.format(str(sampling_technique))),
                                  model_name + '_uncondition')
        inference_cfg.defrost()
        inference_cfg.INPUT.num_empty_tokens_to_ignore = 0
        inference_cfg.OUTPUT.output_txt_directory = output_dir
        inference_cfg.INPUT.num_midi_files = number_of_files
        inference_cfg.INPUT.time_extension = False
        inference_cfg.SAMPLING.technique = 'random'
        inference_cfg.SAMPLING.threshold = topk
        inference_cfg.SAMPLING.temperature = temperature
        inference_cfg.freeze()
        main(inference_cfg)
        cmd = "python3 ../data/music_encoder.py --input_folder {} --output_folder {} --mode " \
              "to_midi".format(output_dir, output_dir + '_MIDI')
        returned_value = subprocess.call(quote(cmd), shell=True)  # returns the exit code in unix

    return returned_value


if __name__ == "__main__":
    args = parse_args()
    inference_cfg = get_default_cfg_inference()
    inference_cfg.merge_from_file(args.inference_config)
    inference_cfg.freeze()

    conditioned_len = 500
    survey_sample = 'batch_samples'
    temperature = 0.95
    temperature_GAN = 0.95
    topk = 32

    prefix_files = glob.glob(os.path.join('../test/', '*.npy'))

    # Condition
    ret = generate_files(inference_cfg, prefix_files, 'topk', topk, 'Baseline', temperature)
    ret = generate_files(inference_cfg, prefix_files, 'random', topk, 'Baseline', temperature)
    # Uncondition
    ret = generate_files(inference_cfg, prefix_files, 'topk', topk, 'Baseline_uncondition', temperature, time_extension=False)
    ret = generate_files(inference_cfg, prefix_files, 'random', topk, 'Baseline_uncondition', temperature, time_extension=False)
