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

from performance_event_repo import PerformanceEventRepo
import functools
import time
import os, sys
import pandas as pd
import logging
sys.path.append(os.path.dirname(sys.path[0]))
from utils import find_files_by_extensions

_CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
MAESTOR_V1_DIR = os.path.join(_CURR_DIR, 'maestro-v1.0.0')

def read_maestro_meta_info(data_dir):
    """Read the meta information from Maestro

    Parameters
    ----------
    data_dir
        The base path of the maestro data

    Returns
    -------
    df
        Pandas Dataframe, with the following columns:
        ['canonical_composer',
         'canonical_title',
         'split',
         'year',
         'midi_filename',
         'audio_filename',
         'duration']
    """
    if os.path.exists(os.path.join(data_dir, 'maestro-v1.0.0.csv')):
        logging.info('Process maestro-v1.')
        csv_path = os.path.join(data_dir, 'maestro-v1.0.0.csv')
    elif os.path.exists(os.path.join(data_dir, 'maestro-v2.0.0.csv')):
        logging.info('Process maestro-v2.')
        csv_path = os.path.join(data_dir, 'maestro-v2.0.0.csv')
    else:
        raise ValueError('Cannot found valid csv files!')
    df = pd.read_csv(csv_path)
    return df


def get_midi_paths():
    if not os.path.exists(MAESTOR_V1_DIR):
        raise ValueError('Cannot find maestro-v1.0.0, use `get_data.sh` to download and '
                         'extract the data.')
    df = read_maestro_meta_info(MAESTOR_V1_DIR)
    train_paths = df[df['split'] == 'train']['midi_filename'].to_numpy()
    validation_paths = df[df['split'] == 'validation']['midi_filename'].to_numpy()
    test_paths = df[df['split'] == 'test']['midi_filename'].to_numpy()
    train_paths = [os.path.join(MAESTOR_V1_DIR, ele) for ele in train_paths]
    validation_paths = [os.path.join(MAESTOR_V1_DIR, ele) for ele in validation_paths]
    test_paths = [os.path.join(MAESTOR_V1_DIR, ele) for ele in test_paths]
    return train_paths, validation_paths, test_paths


if __name__ == '__main__':

    from argparse import ArgumentParser
    import multiprocessing as mpl

    parser = ArgumentParser()
    parser.add_argument('--input_folder', type=str,
                        help='Directory with the downloaded MAESTOR dataset',
                        default=MAESTOR_V1_DIR)
    parser.add_argument('--output_folder', type=str,
                        help='Directory to encode the event signals', required=True)
    parser.add_argument('--encode_official_maestro', action='store_true',
                        help='Whether to encode the official Maestro dataset.')
    parser.add_argument('--mode', type=str,
                        help='Convert to/from MIDIs to TXT/Numpy',
                        choices=['to_txt', 'to_midi',
                                 'midi_to_npy', 'npy_to_midi'],
                        default='to_txt')
    parser.add_argument('--stretch_factors', type=str, help='Stretch Factors',
                        default='0.95,0.975,1.0,1.025,1.05')
    parser.add_argument('--pitch_transpose_lower', type=int,
                        help='Lower bound of the pitch transposition amounts',
                        default=-3)
    parser.add_argument('--pitch_transpose_upper', type=int,
                        help='Uppwer bound of the pitch transposition amounts',
                        default=3)
    args = parser.parse_args()


    stretch_factors = [float(ele) for ele in args.stretch_factors.split(',')]
    encoder = PerformanceEventRepo(steps_per_second=100, num_velocity_bins=32,
                               stretch_factors=stretch_factors,
                               pitch_transpose_lower=args.pitch_transpose_lower,
                               pitch_transpose_upper=args.pitch_transpose_upper)

    def run_to_text(path, out_dir):
        filename, extension = os.path.splitext(os.path.basename(path))
        encoder.to_text(path, os.path.join(out_dir, filename + '.txt'))


    def run_to_text_with_transposition(path, out_dir):
        filename, extension = os.path.splitext(os.path.basename(path))
        encoder.to_text_transposition(path, os.path.join(out_dir, filename + '.txt'))


    def run_to_npy(path, out_dir):
        filename, extension = os.path.splitext(os.path.basename(path))
        encoder.to_npy(path, os.path.join(out_dir, filename + '.npy'))


    def run_to_npy_with_transposition(path, out_dir):
        filename, extension = os.path.splitext(os.path.basename(path))
        encoder.to_npy_transposition(path, os.path.join(out_dir, filename + '.npy'))


    def run_from_text(path, out_dir):
        filename, extension = os.path.splitext(os.path.basename(path))
        encoder.from_text(path, os.path.join(out_dir, filename + '.mid'))


    def run_npy_to_midi(path, out_dir):
        filename, extension = os.path.splitext(os.path.basename(path))
        encoder.npy_to_midi(path, os.path.join(out_dir, filename + '.mid'))

    num_cpus = mpl.cpu_count()
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    if args.mode == 'to_txt' or args.mode == 'midi_to_npy':
        if args.mode == 'to_txt':
            converted_format = 'txt'
            convert_transposition_f = run_to_text_with_transposition
            convert_f = run_to_text
        else:
            converted_format = 'npy'
            convert_transposition_f = run_to_npy_with_transposition
            convert_f = run_to_npy

        print('Converting midi files from {} to {}...'
              .format(args.input_folder, converted_format))
        if args.encode_official_maestro:
            train_paths, valid_paths, test_paths = get_midi_paths()
            print('Load MAESTRO V1 from {}. Train/Val/Test={}/{}/{}'
                  .format(args.input_folder, len(train_paths), len(valid_paths),
                          len(test_paths)))
            for split_name, midi_paths in [('train', train_paths),
                                           ('valid', valid_paths),
                                           ('test', test_paths)]:
                if split_name == 'train':
                    convert_function = convert_transposition_f
                else:
                    convert_function = convert_f
                out_split_dir = os.path.join(args.output_folder, split_name)
                os.makedirs(out_split_dir, exist_ok=True)
                start = time.time()
                with mpl.Pool(num_cpus - 1) as pool:
                    pool.map(functools.partial(convert_function, out_dir=out_split_dir),
                             midi_paths)
                print('Split {} converted! Spent {}s to convert {} samples.'
                      .format(split_name, time.time() - start, len(midi_paths)))
            encoder.create_vocab_txt(args.output_folder)
        else:
            midi_paths = []
            for root, _, files in os.walk(args.input_folder):
                for fname in files:
                    filename, extension = os.path.splitext(os.path.basename(fname))
                    if extension == '.mid' or extension == '.midi':
                        midi_paths.append(os.path.join(root, fname))
            os.makedirs(args.output_folder, exist_ok=True)
            start = time.time()
            with mpl.Pool(num_cpus - 1) as pool:
                pool.map(functools.partial(convert_f, out_dir=args.output_folder),
                         midi_paths)
            print('Converted midi files from {} to {}! Spent {}s to convert {} samples.'
                  .format(args.input_folder, args.output_folder,
                          time.time() - start, len(midi_paths)))
    elif args.mode == 'to_midi' or args.mode == 'npy_to_midi':
        convert_f = run_from_text if args.mode == 'to_midi' else run_npy_to_midi
        start = time.time()
        if args.mode == 'npy_to_midi':
            input_paths = list(find_files_by_extensions(args.input_folder, ['.npy']))
        else:
            input_paths = list(find_files_by_extensions(args.input_folder, ['.txt']))
        with mpl.Pool(num_cpus - 1) as pool:
            pool.map(functools.partial(convert_f,
                                       out_dir=args.output_folder),
                     input_paths)
        print('Test converted! Spent {}s to convert {} samples.'
              .format(time.time() - start, len(input_paths)))
    else:
        raise NotImplementedError
