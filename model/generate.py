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

import argparse
import os
import torch
import numpy as np
import torch.nn.functional as F
from utils.config_inference import get_default_cfg_inference
from utils.config_helper import get_default_cfg_training
from transformer_gan import TransformerGAN
from data_utils import BaseVocab

def load_vocab(vocab_path):
    """
    Loads a vocabulary file

    Parameters
    ----------
    vocab_path : Full file path to the vocabulary file

    Returns
    -------
    :index2token List of the tokens
    :token2index A dictionary mapping tokens to indices 
    """
    tokens_list = []
    with open(vocab_path, "r") as f:
        for line in f:
            tokens_list.append(line.strip())
    token2index = {s: i for i, s in enumerate(tokens_list)}
    return tokens_list, token2index


def parse_args():
    parser = argparse.ArgumentParser(description="DeepComposer Transformer Inference")
    parser.add_argument(
        "--inference_config", type=str, default="inference_config/inference_unconditional.yml", help="path to the cfg file"
    )
    args = parser.parse_args()
    return args


def get_duration_from_token(event_representation, token_index, tokens_list):
    if event_representation == 'magenta':
        conditional_token = tokens_list[token_index]
        if conditional_token.startswith('TIME_SHIFT'):
            duration = int(
                conditional_token.split('_')[-1]) * 0.01  # Each time shift stands 10 ms in magenta event repo
            return duration
        return None
    else:
        raise NotImplementedError


def main(inference_cfg):
    if inference_cfg.EVENT.event_representation == "magenta":
        empty_bar_symbol = "TIME_SHIFT_100"
        if inference_cfg.SAMPLING.technique == "topk":
            if inference_cfg.SAMPLING.threshold:
                topk = int(inference_cfg.SAMPLING.threshold)
            else:
                topk = 32
        elif inference_cfg.SAMPLING.technique == "nucleus":
            if inference_cfg.SAMPLING.threshold:
                p = inference_cfg.SAMPLING.threshold
            else:
                p = 0.95

        elif inference_cfg.SAMPLING.technique == "random":
            topk = 310
    else:
        raise NotImplementedError(
            "Newevent representation generations are yet to be implemented"
        )

    model_fp = os.path.join(inference_cfg.MODEL.model_directory,
                            inference_cfg.MODEL.checkpoint_name)  ## Get the Model Path

    cfg_fp = os.path.join(inference_cfg.MODEL.model_directory, "config.yml")

    if not os.path.isdir(inference_cfg.OUTPUT.output_txt_directory):
        os.makedirs(inference_cfg.OUTPUT.output_txt_directory)
    ext = ".txt"
    device = torch.device("cuda" if inference_cfg.MODEL.device else "cpu")


    tokens_list, token2index = load_vocab(inference_cfg.EVENT.vocab_file_path)
    perform_vocab = BaseVocab(tokens_list)

    nvocab = len(perform_vocab)

    # Encode empty bar token
    empty_bar_token = token2index[empty_bar_symbol]
    # Generate.
    # Params for generation

    cfg = get_default_cfg_training()
    # The following try to merge the configurations from yaml file,
    # and since we have "", which is integrated as None and can not be read by
    # yacs, we have the following block "try: except:" to read from list rather than merge from file.
    try:
        cfg.merge_from_file(cfg_fp)
    except Exception as e:
        print('*' * 100)
        print("Note, if you are loading an old config.yml file which includes None inside,\n"
              " please change it to a string 'None' to make sure you can do cfg.merge_from_file.\n"
              "e.g. cfg.DISCRIMINATOR.type , cfg.TRAIN.pad_type and cfg.TRAIN.load_from_previous.\n"
              "and please note DISCRIMINATOR.temperature is DISCRIMINATOR.beta_max\n")
        print('*' * 100)
        raise e
    cfg.defrost()
    cfg.DISCRIMINATOR.type = "Null"  # cnn for cnn distriminator or Null for no discriminator or 'bert' for BERT
    # discriminator
    cfg.MODEL.same_length = True  # Needed for same_length =True during evaluation
    cfg.freeze()

    if cfg.TRAIN.append_note_status:
        perform_vocab.notes_mapping()

    model = TransformerGAN(cfg, perform_vocab)

    checkpoint = torch.load(model_fp)
    trimmed_checkpoint = {}
    for key, val in checkpoint["model"].items():
        if 'generator' in key:
            new_key = key.replace('generator.', '')
            trimmed_checkpoint[new_key] = val
    model.generator.load_state_dict(trimmed_checkpoint, strict=False)

    # checkpoint = torch.load(model_fp)
    # model.load_state_dict(checkpoint["model"])

    model = model.to(device)
    model.eval()
    model.generator.reset_length(1, inference_cfg.MODEL.memory_length)

    # Load a conditional file for time_extension
    if inference_cfg.INPUT.time_extension:
        conditional_data = np.load(inference_cfg.INPUT.conditional_input_melody).tolist()  # inference_cfg.prefix
        print('* Loaded conditional file {}'.format(inference_cfg.INPUT.conditional_input_melody))
        num_conditional_tokens = inference_cfg.INPUT.num_conditional_tokens
        if inference_cfg.GENERATION.duration_based:
            duration = 0
            for num_conditional_tokens, conditional_index in enumerate(conditional_data):
                token_duration = get_duration_from_token(inference_cfg.EVENT.event_representation, conditional_index,
                                                         tokens_list)
                if token_duration:
                    duration += token_duration  # 10 ms
                if duration >= inference_cfg.INPUT.conditional_duration:
                    break
            # Note, when the conditional duration is longer than the duration of the conditional file,
            # all conditional file will be used.
            print('* Total number of tokens used for condition is {} for duration {}'.format(num_conditional_tokens,
                                                                                             duration))
        else:
            num_conditional_tokens = min(num_conditional_tokens, len(conditional_data))
            print('* Total number of tokens used for condition is {}'.format(num_conditional_tokens))

        with open(os.path.join(inference_cfg.OUTPUT.output_txt_directory, 'prefix' + ext), "w") as f:
            f.write("\n".join(tokens_list[t] for t in conditional_data[:num_conditional_tokens]))
        with open(os.path.join(inference_cfg.OUTPUT.output_txt_directory, 'full' + ext), "w") as f:
            f.write("\n".join(tokens_list[t] for t in conditional_data[:]))


    for midi_file in range(inference_cfg.INPUT.num_midi_files):
        out_fn = str(midi_file) + ext
        out_fp = os.path.join(inference_cfg.OUTPUT.output_txt_directory, out_fn)
        if cfg.TRAIN.replace_start_with_pad:
            seq = [token2index['<PAD>']]  # Pad Token
        else:
            seq = [token2index['<S>']]  # Start Token

        mems = None
        status_vec = None
        with torch.no_grad():
            print("Generating the Midi File Number: " + str(midi_file + 1))
            if inference_cfg.INPUT.time_extension and num_conditional_tokens >= 1:
                # check the argument to do time extension based on a conditional file
                # The time extension model is only activated when inference_cfg.INPUT.time_extension=True
                # and given conditional_len greater than 1
                context = np.array(seq + conditional_data[:num_conditional_tokens - 1], dtype=np.int32)[:, np.newaxis]
                context = torch.from_numpy(context).to(device).type(torch.long)
                if cfg.TRAIN.append_note_status:
                    status_vec = context.new_zeros((context.shape[0], 1, perform_vocab.vec_len), dtype=torch.bool)
                    perform_vocab.update_status_vec(context, status_vec)
                ret = model.generator.forward_generate(context, mems, status_vec=status_vec)
                _, mems = ret
                seq = seq + conditional_data[:num_conditional_tokens]

            if inference_cfg.GENERATION.duration_based:
                duration, generation_length = 0, inference_cfg.GENERATION.max_generation_length
            else:
                generation_length = inference_cfg.GENERATION.generation_length

            for _ in range(generation_length):
                if inference_cfg.GENERATION.duration_based:
                    token_duration = get_duration_from_token(inference_cfg.EVENT.event_representation, seq[-1],
                                                             tokens_list)
                    if token_duration:
                        duration += token_duration
                    if duration >= inference_cfg.GENERATION.generation_duration:
                        break
                # Create input array
                inp = np.array([seq[-1]], dtype=np.int32)[:, np.newaxis]
                inp = torch.from_numpy(inp).to(device).type(torch.long)
                if cfg.TRAIN.append_note_status:
                    bptt, batch_size = inp.shape
                    if status_vec is None:
                        status_vec = inp.new_zeros((bptt, batch_size, perform_vocab.vec_len), dtype=torch.bool)
                    else:
                        status_vec = status_vec[-1:, :, :]
                    perform_vocab.update_status_vec(inp, status_vec)
                ret = model.generator.forward_generate(inp, mems, status_vec=status_vec)
                all_logits, mems = ret
                # Select last timestep, only batch item
                logits = all_logits[-1, 0]

                if inference_cfg.INPUT.exclude_bos_token:
                    logits = logits[1:]

                if inference_cfg.INPUT.num_empty_tokens_to_ignore:
                    # check the total number of empty token (TIME_SHITF_100) in generated sequence
                    # and when the total number of tokens reach a certain number, stops sampling TIME_SHITF_100
                    if np.all(np.asarray(seq[-inference_cfg.INPUT.num_empty_tokens_to_ignore:]) == empty_bar_token):
                        if inference_cfg.INPUT.exclude_bos_token:
                            logits = torch.cat(
                                [logits[:empty_bar_token - 1], logits[empty_bar_token:]], 0
                            )
                        else:
                            logits = torch.cat(
                                [logits[:empty_bar_token], logits[empty_bar_token + 1:]], 0
                            )

                # Handle temp 0 (argmax) case
                if inference_cfg.SAMPLING.temperature == 0:
                    probs = torch.zeros_like(logits)
                    probs[logits.argmax()] = 1.0
                else:
                    # Apply temperature spec
                    logits /= inference_cfg.SAMPLING.temperature

                    # Compute softmax
                    probs = F.softmax(logits, dim=-1)

                if inference_cfg.INPUT.exclude_bos_token:
                    probs = F.pad(probs, [1, 0])

                if inference_cfg.INPUT.num_empty_tokens_to_ignore:
                    if np.all(np.asarray(seq[-inference_cfg.INPUT.num_empty_tokens_to_ignore:]) == empty_bar_token):
                        probs = torch.cat([probs[:empty_bar_token], F.pad(probs[empty_bar_token:], [1, 0])], 0)

                if inference_cfg.SAMPLING.technique == "topk" or inference_cfg.SAMPLING.technique == "random":

                    if inference_cfg.SAMPLING.technique == "random":
                        pass

                    elif topk is not None:
                        _, top_idx = torch.topk(probs, topk)
                        mask = torch.zeros_like(probs)
                        mask[top_idx] = 1.0
                        probs *= mask
                        probs /= probs.sum()

                elif inference_cfg.SAMPLING.technique == "nucleus":
                    if p > 0:
                        sorted_probs, sorted_indices = torch.sort(
                            probs, descending=True
                        )
                        cumulative_probs = torch.cumsum(sorted_probs, dim=0)
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs >= p
                        # Shift the indices to the right to keep also the first token above the threshold

                        sorted_indices_to_remove[1:] = sorted_indices_to_remove[
                                                       :-1
                                                       ].clone()
                        sorted_indices_to_remove[0] = 0
                        # scatter sorted tensors to original indexing
                        indices_to_remove = sorted_indices_to_remove.scatter(
                            dim=0, index=sorted_indices, src=sorted_indices_to_remove
                        )
                        probs[indices_to_remove] = 0
                        probs /= probs.sum()

                else:
                    raise NotImplementedError(
                        "Other Sampling strategies are yet to be implemented"
                    )
                # Sample from probabilities
                token = torch.multinomial(probs, 1)
                token = int(token.item())
                seq.append(token)

            with open(out_fp, "w") as f:
                f.write("\n".join(tokens_list[t] for t in seq[1:]))

            if inference_cfg.MODEL.debug:
                # Ignore last element in seq so that len(mems) is same
                status_vec = None
                data = np.array(seq[:-1], dtype=np.int32)[:, np.newaxis]
                data = torch.from_numpy(data).to(device).type(torch.long)

                if cfg.TRAIN.append_note_status:
                    status_vec = data.new_zeros((data.shape[0], 1, perform_vocab.vec_len), dtype=torch.bool)
                    perform_vocab.update_status_vec(data, status_vec)
                ret = model.generator.forward_generate(data, None, status_vec=status_vec)
                _, new_mems = ret

                assert all(
                    [
                        torch.allclose(i, j, atol=1e-4)
                        for i, j in zip(new_mems, mems)
                    ]
                )
                print("Mem same")
                # This time-shift debug needs to be placed after above
                if inference_cfg.INPUT.time_extension and num_conditional_tokens >= 1:
                    # check the argument to do time extension based on a conditional file
                    # The time extension model is only activated when args.time_extension=True
                    # and given conditional_len greater than 1
                    if cfg.TRAIN.replace_start_with_pad:
                        input_index = token2index['<PAD>']
                    else:
                        input_index = token2index['<S>']
                    nll = 0.
                    status_vec = None
                    for i in range(num_conditional_tokens):
                        target = conditional_data[i]
                        input_index = np.array([input_index], dtype=np.int32)[:, np.newaxis]
                        input_index = torch.from_numpy(input_index).to(device).type(torch.long)
                        if cfg.TRAIN.append_note_status:
                            if status_vec is None:
                                bptt, batch_size = input_index.shape
                                status_vec = inp.new_zeros((bptt, batch_size, perform_vocab.vec_len), dtype=torch.bool)
                            else:
                                status_vec = status_vec[-1:, :, :]
                            perform_vocab.update_status_vec(input_index, status_vec)
                        ret = model.generator.forward_generate(input_index, None,
                                                               status_vec=status_vec)
                        all_logits, _ = ret
                        logits = all_logits[-1, 0]
                        probs = F.softmax(logits, dim=-1)
                        target_prob = probs[target].cpu().item()
                        nll += -np.log(target_prob)
                        input_index = target

                    print('Prime NLL: {}, Prime PPL: {}'.format(nll / num_conditional_tokens,
                                                                np.exp(nll / num_conditional_tokens)))

                with open(os.path.join(inference_cfg.OUTPUT.output_txt_directory, "inference.yml"), "w") as f:
                    f.write(str(inference_cfg))


if __name__ == "__main__":
    args = parse_args()
    inference_cfg = get_default_cfg_inference()
    inference_cfg.merge_from_file(args.inference_config)
    inference_cfg.freeze()
    # Sanity check to make sure the config is indeed right
    print(inference_cfg)
    main(inference_cfg)
