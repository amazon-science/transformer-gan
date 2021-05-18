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
import numpy as np
import torch
import multiprocessing

_CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))


class BaseVocab:
    def __init__(self, all_tokens):
        self._all_tokens = all_tokens
        self._map = dict()
        self._reverse_map = dict()
        for i, token in enumerate(all_tokens):
            self._map[token] = i
            self._reverse_map[i] = token
        assert self._all_tokens[0] == "<S>"
        assert self._all_tokens[1] == "<PAD>"
        self.vec_len = 0

    def idx_to_token(self, idx):
        return self._all_tokens[idx]

    @property
    def bos_token(self):
        return self._all_tokens[0]

    @property
    def pad_token(self):
        return self._all_tokens[1]

    @property
    def bos_id(self):
        return 0

    @property
    def pad_id(self):
        return 1

    @property
    def all_tokens(self):
        return self._all_tokens

    def token_to_idx(self, token):
        return self._map[token]

    def __len__(self):
        return len(self._all_tokens)

    def __getitem__(self, token):
        return self._map[token]

    def notes_mapping(self):
        # Maps note on and note off tokens to a vector for append_note_status
        # functionality
        # Assume note_on and note_off are in sequence
        note_on_tokens = [i for i in self._map.keys() if 'NOTE_ON' in i]
        note_off_tokens = [i for i in self._map.keys() if 'NOTE_OFF' in i]

        self.vec_len = len(note_on_tokens)
        self.note_on_dic = dict()
        self.note_off_dic = dict()
        index = 0
        for note_on, note_off in zip(note_on_tokens, note_off_tokens):
            self.note_on_dic[self._map[note_on]] = index
            self.note_off_dic[self._map[note_off]] = index
            index += 1

    def update_status_vec(self, data, status_vec):
        bptt, bsz = data.shape

        # data_npy = data.numpy()
        with torch.no_grad():
            for batch in range(bsz):
                temp = status_vec[-1, batch, :].clone()
                for token in range(bptt):
                    val = data[token, batch].item()
                    if val in self.note_on_dic:
                        temp[self.note_on_dic[val]] = True
                    elif val in self.note_off_dic:
                        temp[self.note_off_dic[val]] = False
                    else:
                        pass
                    status_vec[token, batch, :] = temp

class MusicDataset:
    def __init__(self, data_dir, cfg):
        """Load the music corpus
        Args:
            data_dir: The base folder of the preprocessed music dataset
        """
        self._vocab_path = os.path.join(data_dir, "vocab.txt")
        self._train_folder = os.path.join(data_dir, "train")
        self._valid_folder = os.path.join(data_dir, "valid")
        self._test_folder = os.path.join(data_dir, "test")
        all_tokens = []
        with open(self._vocab_path, "r") as f:
            for token in f:
                token = token.strip()
                all_tokens.append(token)
        self._vocab = BaseVocab(all_tokens)

        self._train_data = self.load_cache_data(self._train_folder)
        self._valid_data = self.load_cache_data(self._valid_folder)
        self._test_data = self.load_cache_data(self._test_folder)
        self.cfg = cfg

        # Insert start tokens
        if self.cfg.TRAIN.replace_start_with_pad:
            print("USING PAD TOKEN AS START!")
            insert_token = self._vocab.pad_id
        else:
            insert_token = self._vocab.bos_id
        self._train_data = [
            torch.from_numpy(np.insert(arr, 0, insert_token))
            for arr in self._train_data
        ]
        self._valid_data = [
            torch.from_numpy(np.insert(arr, 0, insert_token))
            for arr in self._valid_data
        ]
        self._test_data = [
            torch.from_numpy(np.insert(arr, 0, insert_token))
            for arr in self._test_data
        ]

        self._train_seq_length = np.array(
            [ele.shape[0] for ele in self._train_data], dtype=np.int32
        )
        self._valid_seq_length = np.array(
            [ele.shape[0] for ele in self._valid_data], dtype=np.int32
        )
        self._test_seq_length = np.array(
            [ele.shape[0] for ele in self._test_data], dtype=np.int32
        )
        print(
            "Loaded Data, #Samples Train/Val/Test:{}/{}/{}".format(
                len(self._train_data), len(self._valid_data), len(self._test_data)
            )
        )
        print(
            "             #Avg Length:{}/{}/{}".format(
                np.mean([len(ele) for ele in self._train_data]),
                np.mean([len(ele) for ele in self._valid_data]),
                np.mean([len(ele) for ele in self._test_data]),
            )
        )
        print(
            "             #Total Number of Valid/Test Tokens: {}/{}".format(
                (self._valid_seq_length - 1).sum(), (self._test_seq_length - 1).sum()
            )
        )
        if cfg.TRAIN.append_note_status:
            self._vocab.notes_mapping()



    def load_cache_data(self, dir_name):
        all_fnames = sorted(glob.glob(os.path.join(dir_name, "*.npy")))
        print("Loading #{} files from {}".format(len(all_fnames), dir_name))
        # We will create a large array
        with multiprocessing.Pool(8) as pool:
            dat = pool.map(np.load, all_fnames)
        return np.array(dat)

    @property
    def vocab(self):
        return self._vocab

    @property
    def train_data(self):
        return self._train_data

    @property
    def valid_data(self):
        return self._valid_data

    @property
    def test_data(self):
        return self._test_data

    @property
    def train_seq_length(self):
        return self._train_seq_length

    @property
    def valid_seq_length(self):
        return self._valid_seq_length

    @property
    def test_seq_length(self):
        return self._test_seq_length


    def get_iterator(
            self, batch_size, bptt, device, split="train", do_shuffle=True, seed=None
    ):
        if split == "train":
            split_data = self.train_data
            split_seq_lengths = self.train_seq_length
        elif split == "valid":
            split_data = self.valid_data
            split_seq_lengths = self.valid_seq_length
        elif split == "test":
            split_data = self.test_data
            split_seq_lengths = self.test_seq_length
        else:
            raise NotImplementedError
        total_sample_num = len(split_data)

        def iterator():
            perm = np.arange(total_sample_num)
            if do_shuffle:
                rng = np.random.RandomState(seed)
                rng.shuffle(perm)
            assert batch_size < total_sample_num
            tracker_list = [(i, 0) for i in range(batch_size)]
            next_idx = batch_size
            data = torch.LongTensor(bptt, batch_size)
            target = torch.LongTensor(bptt, batch_size)
            reset_mem = torch.BoolTensor(batch_size)

            if self.cfg.TRAIN.append_note_status:
                status_vec = torch.zeros((bptt, batch_size, self._vocab.vec_len), dtype=torch.bool)
            else:
                status_vec = None

            while True:
                # Generate the samples
                # Fill with pad_id
                data[:] = self.vocab.pad_id
                target[:] = self.vocab.pad_id
                reset_mem[:] = False
                batch_token_num = 0
                for i in range(batch_size):
                    idx, pos = tracker_list[i]
                    while idx < total_sample_num:
                        seq_id = perm[idx]
                        seq_length = split_seq_lengths[seq_id]
                        if pos + 1 >= seq_length:
                            idx, pos = next_idx, 0
                            tracker_list[i] = (idx, pos)
                            next_idx += 1
                            reset_mem[i] = True
                            continue
                        else:
                            if self.cfg.TRAIN.random_crop:
                                offset = 0
                                if self.cfg.TRAIN.mem_length == 0:
                                    offset = bptt
                                if pos == 0:
                                    # print("Picking random span")
                                    pos = np.random.randint(0, seq_length - 1 - offset)  # Atleast bptt

                            n_new = min(seq_length - 1 - pos, bptt)
                            data[:n_new, i] = split_data[seq_id][pos: pos + n_new]
                            target[:n_new, i] = split_data[seq_id][
                                                (pos + 1): (pos + 1 + n_new)]
                            batch_token_num += n_new
                            tracker_list[i] = (idx, pos + n_new)

                            if self.cfg.TRAIN.mem_length == 0 and self.cfg.TRAIN.random_crop:
                                # Move on if memlen==0
                                idx, pos = next_idx, 0
                                tracker_list[i] = (idx, pos)
                                next_idx += 1
                                reset_mem[i] = True

                            break
                if batch_token_num == 0:
                    # Haven't found anything to fill. This indicates we have reached the end
                    if do_shuffle:
                        rng.shuffle(perm)
                    else:
                        return  # One pass dataloader when do_shuffle is False
                    tracker_list = [(i, 0) for i in range(batch_size)]
                    next_idx = batch_size
                    continue

                if self.cfg.TRAIN.append_note_status:
                    # Reset status vec for new midi file
                    status_vec[:, reset_mem, :] = False
                    self._vocab.update_status_vec(data, status_vec)
                    status_vec = status_vec.to(device)

                yield data.to(device), target.to(device), reset_mem.to(
                    device), batch_token_num, status_vec

        return iterator


    def get_dis_iterator(
            self, batch_size, bptt, device, split="train", do_shuffle=True, seed=None
    ):
        # Iterator selects random chunk of length bptt from each midi
        if split == "train":
            split_data = self.train_data
            split_seq_lengths = self.train_seq_length
        # dis val uses different permutation so this is okay
        elif split == "valid":
            split_data = self.valid_data
            split_seq_lengths = self.valid_seq_length
        elif split == "test":
            split_data = self.test_data
            split_seq_lengths = self.test_seq_length
        else:
            raise ValueError

        total_sample_num = len(split_data)

        def iterator():
            perm = np.arange(total_sample_num)
            if do_shuffle:
                rng = np.random.RandomState(seed)
                rng.shuffle(perm)
            assert batch_size < total_sample_num
            tracker_list = [(i, 0) for i in range(batch_size)]
            next_idx = batch_size
            data = torch.LongTensor(bptt, batch_size)
            while True:
                # Generate the samples
                # Fill with pad_id
                data[:] = self.vocab.pad_id
                batch_token_num = 0
                for i in range(batch_size):
                    idx, pos = tracker_list[i]
                    while idx < total_sample_num:
                        seq_id = perm[idx]
                        seq_length = split_seq_lengths[seq_id]
                        if bptt > seq_length:
                            idx, pos = next_idx, 0
                            tracker_list[i] = (idx, pos)
                            next_idx += 1
                            continue
                        else:
                            # Fill elements
                            pos = np.random.randint(0, seq_length - bptt + 1)
                            data[:bptt, i] = split_data[seq_id][pos: pos + bptt]
                            batch_token_num += bptt
                            tracker_list[i] = (idx, pos + bptt)
                            break
                if batch_token_num == 0:
                    # Haven't found anything to fill. This indicates we have reached the end
                    if do_shuffle:
                        rng.shuffle(perm)
                    else:
                        return  # One pass dataloader when do_shuffle is False
                    tracker_list = [(i, 0) for i in range(batch_size)]
                    next_idx = batch_size
                    continue

                yield data.to(device), batch_token_num

        return iterator

    def eval_iterator(
            self, batch_size, bptt, device, split="valid", local_rank=0, world_size=0
    ):
        if split == "valid":
            split_data = self.valid_data
            split_seq_lengths = self.valid_seq_length
        elif split == "test":
            split_data = self.test_data
            split_seq_lengths = self.test_seq_length
        else:
            raise NotImplementedError
        if world_size > 0:
            all_sample_num = len(split_data)
            if local_rank == world_size - 1:
                begin_idx = all_sample_num // world_size * local_rank
                end_idx = all_sample_num
            else:
                begin_idx = all_sample_num // world_size * local_rank
                end_idx = all_sample_num // world_size * (local_rank + 1)
            split_data = split_data[begin_idx:end_idx]
            split_seq_lengths = split_seq_lengths[begin_idx:end_idx]
        total_sample_num = len(split_data)

        def iterator():
            data = torch.LongTensor(bptt, batch_size)
            target = torch.LongTensor(bptt, batch_size)
            if self.cfg.TRAIN.append_note_status:
                status_vec = torch.zeros((bptt, batch_size, self._vocab.vec_len), dtype=torch.bool)
            else:
                status_vec = None
            for batch_begin in range(0, total_sample_num, batch_size):
                reset_all_mem = True
                batch_end = min(batch_begin + batch_size, total_sample_num)
                max_seq_length = max(split_seq_lengths[batch_begin:batch_end])
                for seq_begin in range(0, max_seq_length - 1, bptt):
                    data[:] = self.vocab.pad_id
                    target[:] = self.vocab.pad_id
                    batch_token_num = 0
                    for i in range(batch_begin, batch_end):
                        if split_seq_lengths[i] > seq_begin + 1:
                            n_new = (
                                    min(seq_begin + bptt, split_seq_lengths[i] - 1)
                                    - seq_begin
                            )
                            data[:n_new, i - batch_begin] = split_data[i][
                                                            seq_begin: seq_begin + n_new
                                                            ]
                            target[:n_new, i - batch_begin] = split_data[i][
                                                              (seq_begin + 1): (seq_begin + n_new + 1)
                                                              ]
                            batch_token_num += n_new

                    if self.cfg.TRAIN.append_note_status:
                        # Reset status vec for new midi file
                        if reset_all_mem:
                            status_vec[:] = False
                        self._vocab.update_status_vec(data, status_vec)
                        status_vec = status_vec.to(device)

                    yield data.to(device), target.to(
                        device), reset_all_mem, batch_token_num, status_vec

                    reset_all_mem = False

        return iterator


if __name__ == "__main__":
    from utils.config_helper import get_default_cfg_training

    cfg = get_default_cfg_training()

    data_dir = os.path.join(_CURR_DIR, "..", "data", "maestro_magenta_s5_t3")
    dataset = MusicDataset(data_dir, cfg)
    train_iter = dataset.get_iterator(8, 32, torch.device("cpu"), "train", True)
    valid_iter = dataset.get_iterator(8, 32, torch.device("cpu"), "valid", False)
    test_iter = dataset.get_iterator(8, 32, torch.device("cpu"), "test", False)
    valid_eval_iter = dataset.eval_iterator(8, 32, torch.device("cpu"), "valid")
    test_eval_iter = dataset.eval_iterator(8, 32, torch.device("cpu"), "test")
    seq_lengths = np.zeros((8,))
    stop = False
    for data, target, reset_mem, batch_token_num, status_vec in train_iter():
        reset_mem = reset_mem.cpu().numpy()
        assert (target != dataset.vocab.pad_id).sum().item() == batch_token_num
        for j in range(8):
            seq_lengths[j] += (data[:, j] != dataset.vocab.pad_id).sum().cpu().numpy()
            if reset_mem[j]:
                seq_lengths[j] = 0
                stop = True
        if stop:
            break

    total_val_token_num = 0
    for i, (data, target, reset_mem, batch_token_num, status_vec) in enumerate(valid_eval_iter()):
        assert (target != dataset.vocab.pad_id).sum().item() == batch_token_num
        total_val_token_num += batch_token_num
    assert total_val_token_num == (dataset.valid_seq_length - 1).sum()
    total_val_token_num = 0
    for i, (data, target, reset_mem, batch_token_num, status_vec) in enumerate(valid_iter()):
        assert (target != dataset.vocab.pad_id).sum().item() == batch_token_num
        total_val_token_num += batch_token_num
    assert total_val_token_num == (dataset.valid_seq_length - 1).sum()

    total_test_token_num = 0
    for i, (data, target, reset_mem, batch_token_num, status_vec) in enumerate(test_eval_iter()):
        assert (target != dataset.vocab.pad_id).sum().item() == batch_token_num
        total_test_token_num += batch_token_num
    assert total_test_token_num == (dataset.test_seq_length - 1).sum()

    total_test_token_num = 0
    for i, (data, target, reset_mem, batch_token_num, status_vec) in enumerate(test_iter()):
        assert (target != dataset.vocab.pad_id).sum().item() == batch_token_num
        total_test_token_num += batch_token_num
    assert total_test_token_num == (dataset.test_seq_length - 1).sum()

    # Test for validation distributed iterator
    eval_iter_l = [
        dataset.eval_iterator(
            8, 32, torch.device("cpu"), "valid", local_rank=i, world_size=8
        )
        for i in range(8)
    ]
    total_val_token_num = 0
    for eval_iter in eval_iter_l:
        for i, (data, target, reset_mem, batch_token_num, status_vec) in enumerate(eval_iter()):
            assert (target != dataset.vocab.pad_id).sum().item() == batch_token_num
            total_val_token_num += batch_token_num
    assert total_val_token_num == (dataset.valid_seq_length - 1).sum()

    # Test for testing distributed iterator
    test_iter_l = [
        dataset.eval_iterator(
            8, 32, torch.device("cpu"), "test", local_rank=i, world_size=8
        )
        for i in range(8)
    ]
    total_test_token_num = 0
    for test_iter in test_iter_l:
        for i, (data, target, reset_mem, batch_token_num, status_vec) in enumerate(test_iter()):
            assert (target != dataset.vocab.pad_id).sum().item() == batch_token_num
            total_test_token_num += batch_token_num
    assert total_test_token_num == (dataset.test_seq_length - 1).sum()
