# The following code is based on
# https://github.com/Tiiiger/bert_score/blob/df3108064ca4243a94d0b67f292503ff806ba8b0/bert_score/score.py
# https://github.com/Tiiiger/bert_score/blob/df3108064ca4243a94d0b67f292503ff806ba8b0/bert_score/utils.py

# Tiiiger/bert_score is licensed under the MIT License
# Copyright (c) 2019 Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi.
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import torch
from typing import Dict
from os import listdir
import argparse
import logging
import os
import glob
from tokenization_midi import MIDITokenizer as BertTokenizer
import collections
import time
import sys

from transformers import (
    BertConfig,
    BertForMaskedLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)

block_size = 512

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logging.info("Running BERT")
logger = logging.getLogger('BERT')  # __name__
logger.setLevel(20)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
}


def sent_encode(tokenizer, path, len_tokens_evaluated=2048):
    # "Encoding as sentence based on the tokenizer"
    examples = []
    tokenized_text = tokenizer.encode(path)[:len_tokens_evaluated]
    for i in range(0, len(tokenized_text) - block_size + 1, block_size):
        examples.append(tokenized_text[i: i + block_size])
    return examples


def bert_encode(model, x, attention_mask, all_layers=False):
    model.eval()
    with torch.no_grad():
        out = model(x, attention_mask=attention_mask)  # prediction_score
    if all_layers:
        emb = torch.stack(out[-1], dim=2)
    else:
        emb = out[0]
    return emb


def get_bert_embedding(path, arr, model, tokenizer, all_layers=False):
    """
    Compute BERT embedding in batches.
    Args:
        - :param: `all_sens` (list of str) : sentences to encode.
        - :param: `model` : a BERT model from `pytorch_pretrained_bert`.
        - :param: `tokenizer` : a BERT tokenizer corresponds to `model`.
        - :param: `idf_dict` (dict) : mapping a word piece index to its
                               inverse document frequency
        - :param: `device` (str): device to use, e.g. 'cpu' or 'cuda'
    """
    print('working in', path)
    max_len = max([len(a) for a in arr])
    attention_mask = np.zeros((len(arr), max_len), dtype=np.float)
    for i, a in enumerate(arr):
        if len(a) < max_len:
            arr[i] = a + [tokenizer.pad_token_id] * (max_len - len(a))
            attention_mask[i, len(a):] = 1
    arr = np.array(arr, dtype=np.int)
    m = torch.nn.LogSoftmax(dim=2)
    sub_batch_size = 256
    sum_ = 0
    with torch.no_grad():
        for i in range(0, len(arr)):
            x = arr[i:i + 1]
            # repeat x
            x_repeat_seqlen = np.repeat(x, block_size, axis=0).reshape(-1, x.shape[-1])
            x_repeat_seqlen[np.arange(block_size), np.arange(block_size)] = tokenizer.mask_token_id
            x_repeat_seqlen_cuda = torch.from_numpy(x_repeat_seqlen).to('cuda')

            # repeat mask
            mask_repeat_seqlen = torch.from_numpy(np.repeat(attention_mask[i:i + 1], block_size, axis=0).reshape(-1,
                                                                                                                 attention_mask[
                                                                                                                 i:i 
                                                                                                                   + 
                                                                                                                   1].shape[
                                                                                                                     -1])).to(
                'cuda')

            overal_likelihood = []
            for j in range(0, x_repeat_seqlen.shape[0], sub_batch_size):
                batch_prediction_score = bert_encode(model, x_repeat_seqlen_cuda[j:j + sub_batch_size],
                                                     attention_mask=mask_repeat_seqlen[j:j + sub_batch_size],
                                                     all_layers=all_layers)
                likelihood = m(batch_prediction_score)
                overal_likelihood.append(likelihood)

            likelihood = torch.cat(overal_likelihood, 0)
            likelihood = likelihood.type(torch.float16)
            likelihood_np = likelihood.cpu().clone().numpy()
            sequence_len, _, vocab_size = likelihood_np.shape
            tmp = likelihood_np[np.arange(sequence_len), np.arange(sequence_len), np.array(x[0, :])].mean()
            sum_ += tmp

    return (sum_ / len(arr))


def run_score(model, tokenizer, len_tokens_evaluated=2048):
    subfolders = [f for f in listdir("inference")]
    models_likelihood = {}
    model_likelihood = collections.defaultdict(list)

    for modelname in subfolders:
        model_path = os.path.join('inference', modelname)
        model_files = glob.glob(os.path.join(model_path, '*.npy'))
        arrs = []
        for path in model_files:
            arr = sent_encode(tokenizer, path, len_tokens_evaluated)
            arrs.append(arr)

        for i, arr in enumerate(arrs):
            likelihood = get_bert_embedding(model_files[i], arr, model, tokenizer, all_layers=False)
            model_likelihood[modelname].append(likelihood)

        values = model_likelihood[modelname]
        if len(values) >= 1:
            models_likelihood[modelname] = (np.mean(values), np.var(values))
            with open('result_{}.txt'.format(modelname), 'w') as f:
                print(models_likelihood, file=f)

    print('-------------------------')
    print('Result: ')
    print(models_likelihood)
    with open('result.txt', 'w') as f:
        print(models_likelihood, file=f)


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )
    parser.add_argument('--fp16', default=True,
                        help='Run in pseudo-fp16 mode (fp16 storage fp32 math).')

    parser.add_argument(
        "--model_type", type=str, default='bert', help="The model architecture to be trained or fine-tuned.",
    )

    parser.add_argument(
        "--vocab_file",
        type=str,
        required=True,
        help="The vocab file.",
    )

    parser.add_argument(
        "--event_type",
        type=str,
        required=True,
        help="The event type.",
        choices=['magenta', 'newevent']
    )

    parser.add_argument(
        "--len_tokens_evaluated",
        type=int,
        default=2048,
        help="Total max number of tokens to be evaluated.",
    )

    args = parser.parse_args()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path, cache_dir=None)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=None)

    if args.vocab_file:
        tokenizer.build_vocab_file(args.vocab_file, event_type=args.event_type)

    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=None,
    )

    if args.fp16:
        model = model.half()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    run_score(model, tokenizer, args.len_tokens_evaluated)
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
