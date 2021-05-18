from data_utils import MusicDataset
import os, torch
import numpy as np

_CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))


def test_data_utils():
    data_dir = os.path.join(_CURR_DIR, '..', 'data', 'maestro_magenta_s5_t3')
    dataset = MusicDataset(data_dir)
    BATCH_SIZE, BPTT = 16, 32
    train_iter = dataset.get_iterator(BATCH_SIZE, BPTT, torch.device('cpu'), 'train', True)
    valid_iter = dataset.get_iterator(BATCH_SIZE, BPTT, torch.device('cpu'), 'valid', False)
    test_iter = dataset.get_iterator(BATCH_SIZE, BPTT, torch.device('cpu'), 'test', False)
    valid_eval_iter = dataset.eval_iterator(BATCH_SIZE, BPTT, torch.device('cpu'), 'valid')
    test_eval_iter = dataset.eval_iterator(BATCH_SIZE, BPTT, torch.device('cpu'), 'test')
    seq_lengths = np.zeros((BATCH_SIZE,))
    stop = False
    for data, target, reset_mem, batch_token_num in train_iter():
        reset_mem = reset_mem.cpu().numpy()
        assert (target != dataset.vocab.pad_id).sum().item() == batch_token_num
        for j in range(BATCH_SIZE):
            seq_lengths[j] += (data[:, j] != dataset.vocab.pad_id).sum().cpu().numpy()
            if reset_mem[j]:
                seq_lengths[j] = 0
                stop = True  # Run one pass of the dataset
        if stop:
            break

    total_val_token_num = 0
    for i, (data, target, reset_mem, batch_token_num) in enumerate(valid_eval_iter()):
        assert (target != dataset.vocab.pad_id).sum().item() == batch_token_num
        total_val_token_num += batch_token_num
    assert total_val_token_num == (dataset.valid_seq_length - 1).sum()
    total_val_token_num = 0
    for i, (data, target, reset_mem, batch_token_num) in enumerate(valid_iter()):
        assert (target != dataset.vocab.pad_id).sum().item() == batch_token_num
        total_val_token_num += batch_token_num
    assert total_val_token_num == (dataset.valid_seq_length - 1).sum()

    total_test_token_num = 0
    for i, (data, target, reset_mem, batch_token_num) in enumerate(test_eval_iter()):
        assert (target != dataset.vocab.pad_id).sum().item() == batch_token_num
        total_test_token_num += batch_token_num
    assert total_test_token_num == (dataset.test_seq_length - 1).sum()

    total_test_token_num = 0
    for i, (data, target, reset_mem, batch_token_num) in enumerate(test_iter()):
        assert (target != dataset.vocab.pad_id).sum().item() == batch_token_num
        total_test_token_num += batch_token_num
    assert total_test_token_num == (dataset.test_seq_length - 1).sum()

    # Test for validation distributed iterator
    eval_iter_l = [dataset.eval_iterator(BATCH_SIZE, BPTT, torch.device('cpu'), 'valid',
                                         local_rank=i, world_size=8) for i in range(8)]
    total_val_token_num = 0
    for eval_iter in eval_iter_l:
        for i, (data, target, reset_mem, batch_token_num) in enumerate(eval_iter()):
            assert (target != dataset.vocab.pad_id).sum().item() == batch_token_num
            total_val_token_num += batch_token_num
    assert total_val_token_num == (dataset.valid_seq_length - 1).sum()

    # Test for testing distributed iterator
    test_iter_l = [dataset.eval_iterator(BATCH_SIZE, BPTT, torch.device('cpu'), 'test',
                                         local_rank=i, world_size=8) for i in range(8)]
    total_test_token_num = 0
    for test_iter in test_iter_l:
        for i, (data, target, reset_mem, batch_token_num) in enumerate(test_iter()):
            assert (target != dataset.vocab.pad_id).sum().item() == batch_token_num
            total_test_token_num += batch_token_num
    assert total_test_token_num == (dataset.test_seq_length - 1).sum()
