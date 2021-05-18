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
import time
import math
import os
import numpy as np
import contextlib
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import lamb


from utils.config_helper import get_default_cfg_training
from torch.nn.parallel import DistributedDataParallel as DDP

# from apex.parallel import DistributedDataParallel as DDP
from data_utils import MusicDataset
from utils.exp_utils import logging_config
import torch.distributed as dist
from utils.helpers import get_fixed_temperature



from transformers import (
    BertConfig,
    BertForMaskedLM,
    PreTrainedTokenizer,
    PreTrainedModel,
    AdamW,
)

from utils.bleu import BLEU
from utils.classifier import Classifier

# TODO: can we move handling of mems to inside transformer_gan?
from transformer_gan import TransformerGAN


# from torch.utils.tensorboard import SummaryWriter
# default `log_dir` is "runs" - we'll be more specific here
# writer = SummaryWriter('gumbel_experiment_1')
# try:
#     from torch.utils.tensorboard import SummaryWriter
# except ImportError:
#     from tensorboardX import SummaryWriter

# Define default cfg (for model related params)


@contextlib.contextmanager
def sync_workers(args):
    """
    Yields distributed rank and synchronizes all workers on exit.
    """
    yield args.local_rank
    dist.barrier()


def save_checkpoint(
        args,
        model,
        optimizer,
        dis_optimizer,
        gen_optimizer,
        vocab,
        train_step,
        best_val_loss,
        scheduler,
        dis_scheduler,
        gen_scheduler,
        name="checkpoint.pt",
):
    checkpoint = {
        "model": model.module.state_dict(),  # non ddp checkpoint
        "optimizer": optimizer.state_dict(),
        "train_step": train_step,
        "scheduler": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "vocab": vocab,
    }
    if dis_optimizer:
        checkpoint["dis_optimizer"] = dis_optimizer.state_dict()
    if dis_scheduler:
        checkpoint["dis_scheduler"] = dis_scheduler.state_dict()
    if gen_optimizer:
        checkpoint["gen_optimizer"] = gen_optimizer.state_dict()
    if gen_scheduler:
        checkpoint["gen_scheduler"] = gen_scheduler.state_dict()

    if args.fp16:
        checkpoint["amp"] = amp.state_dict()
    else:
        checkpoint["amp"] = None

    with sync_workers(args) as rank:
        path = os.path.join(args.work_dir, name)
        logging.info(f"Saving checkpoint to {path}")
        if rank == 0:
            torch.save(checkpoint, path)


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Transformer Language Model")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="location of the data corpus"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--work_dir",
        type=str,
        required=True,
        help="Base directory to save the trained model.",
    )
    parser.add_argument("--fp16", action="store_true", help="Whether to use fp16")
    parser.add_argument(
        "--cfg", type=str, default="transformer_xl.yml", help="path to the cfg file"
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Whether to restart from the existing checkpoint",
    )
    parser.add_argument("--debug", action="store_true", help="Debug the program.")
    parser.add_argument("--save-all", action="store_true", help="Save all checkpoints")
    # parser.add_argument('--restart_dir', type=str, default='', help='restart dir')
    args = parser.parse_args()
    return args


args = parse_args()
cfg = get_default_cfg_training()
cfg.merge_from_file(args.cfg)
torch.cuda.set_device(args.local_rank)
device = torch.device("cuda", args.local_rank)
torch.distributed.init_process_group(backend="nccl", init_method="env://")

exp_time = torch.tensor(time.time(), dtype=torch.float64).to(device)
torch.distributed.broadcast(exp_time, 0)
exp_time = float(exp_time.cpu().numpy())

if not args.restart:
    args.work_dir = os.path.join(
        args.work_dir, time.strftime("%Y%m%d-%H%M%S", time.localtime(exp_time))
    )

    os.makedirs(args.work_dir, exist_ok=True)

    # Save necessary files
    if args.local_rank == 0:
        with open(os.path.join(args.work_dir, "config.yml"), "w") as f:
            f.write(str(cfg))

# Save necessary files
if args.local_rank == 0:
    logging_config(args.work_dir, "train_rank{}".format(args.local_rank), console=True)
else:
    logging_config(args.work_dir, "train_rank{}".format(args.local_rank), console=False)

# Set the random seed manually for reproducibility.
seed = cfg.TRAIN.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Validate `--fp16` option
if args.fp16:
    try:
        import apex
        from apex import amp

        apex.amp.register_half_function(torch, "einsum")
    except:
        raise ImportError("WARNING: apex not installed, ignoring --fp16 option")

###############################################################################
# Create dis config
###############################################################################
if (
        cfg.DISCRIMINATOR.type != "bert"
        and cfg.DISCRIMINATOR.type != "cnn"
        and cfg.DISCRIMINATOR.type != "Null"
        and cfg.DISCRIMINATOR.type != ""
):
    raise NotImplementedError

# if cfg.DISCRIMINATOR.freeze_discriminator == True and cfg.DISCRIMINATOR.type != "bert":
#     raise NotImplementedError

if (
        cfg.DISCRIMINATOR.type == "Null" or cfg.DISCRIMINATOR.type == ""
) and cfg.DISCRIMINATOR.start_iter < cfg.TRAIN.max_step:
    raise ValueError

if cfg.DISCRIMINATOR.sample_chunks_mem > 1:
    assert cfg.DISCRIMINATOR.tgt_len % cfg.DISCRIMINATOR.sample_chunks_mem == 0

assert (
        cfg.DISCRIMINATOR.context_len >= 1
        and cfg.DISCRIMINATOR.context_len <= cfg.DISCRIMINATOR.tgt_len
)

###############################################################################
# Load data
###############################################################################
print("Loading data")
dataset = MusicDataset(args.data_dir, cfg)
vocab = dataset.vocab

local_seed = cfg.TRAIN.seed + args.local_rank * 1000
num_gpus = torch.cuda.device_count()
assert cfg.TRAIN.batch_size % num_gpus == 0
batch_size = cfg.TRAIN.batch_size // num_gpus

train_iter = dataset.get_iterator(
    batch_size, cfg.TRAIN.tgt_length, device, "train", True, seed=local_seed
)
val_iter = dataset.eval_iterator(
    cfg.EVALUATE.batch_size,
    cfg.EVALUATE.tgt_length,
    device,
    "valid",
    local_rank=args.local_rank,
    world_size=num_gpus,
)
test_iter = dataset.eval_iterator(
    cfg.EVALUATE.batch_size,
    cfg.EVALUATE.tgt_length,
    device,
    "test",
    local_rank=args.local_rank,
    world_size=num_gpus,
)
if cfg.DISCRIMINATOR.type != "Null" and cfg.DISCRIMINATOR.type != "":
    dis_iter = dataset.get_dis_iterator(
        batch_size, cfg.DISCRIMINATOR.tgt_len, device, "train", True, seed=local_seed,
    )
    # dis_val_iter = dataset.get_dis_iterator(
    #     cfg.EVALUATE.batch_size,
    #     cfg.DISCRIMINATOR.tgt_len,
    #     device,
    #     "valid",
    #     True,
    #     seed=local_seed,
    # )
    # dis_test_iter = dataset.get_dis_iterator(
    #     cfg.EVALUATE.batch_size,
    #     cfg.DISCRIMINATOR.tgt_len,
    #     device,
    #     "test",
    #     True,
    #     seed=local_seed,
    # )
else:
    dis_iter = None


###############################################################################
# Define metrics
###############################################################################

bleu = BLEU("BLEU", gram=[2, 3, 4, 5], if_use=cfg.METRICS.use_bleu)
self_bleu = BLEU("Self-BLEU", gram=[2, 3, 4], if_use=cfg.METRICS.use_self_bleu)
classifier = Classifier("Classifier", if_use=cfg.METRICS.CLASSIFIER.use_classifier, device=device,
                        seq_len=cfg.METRICS.CLASSIFIER.block_size, batch_size=cfg.METRICS.CLASSIFIER.bert_batch_size,
                        model_name_or_path=cfg.METRICS.CLASSIFIER.model_path)
eval_metrics = [bleu, self_bleu, classifier]
test_metrics = [bleu]

###############################################################################
# Build the model
###############################################################################

print("Build the model")


def init_weight(weight):
    if cfg.INITIALIZER.base_init[0] == "normal":
        init_std = cfg.INITIALIZER.base_init[1]
        nn.init.normal_(weight, 0.0, init_std)
    elif cfg.INITIALIZER.base_init[0] == "uniform":
        init_range = cfg.INITIALIZER.base_init[1]
        nn.init.uniform_(weight, -init_range, init_range)


def init_embed(weight):
    if cfg.INITIALIZER.embed_init[0] == "normal":
        init_std = cfg.INITIALIZER.embed_init[1]
        nn.init.normal_(weight, 0.0, init_std)
    elif cfg.INITIALIZER.embed_init[0] == "uniform":
        init_range = cfg.INITIALIZER.embed_init[1]
        nn.init.uniform_(weight, -init_range, init_range)


def init_bias(bias):
    nn.init.constant_(bias, 0.0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
            init_bias(m.bias)
    elif classname.find("AdaptiveEmbedding") != -1:
        if hasattr(m, "emb_projs"):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    init_embed(m.emb_projs[i])
    elif classname.find("Embedding") != -1:
        if hasattr(m, "weight"):
            init_weight(m.weight)
    elif classname.find("ProjectedAdaptiveLogSoftmax") != -1:
        if hasattr(m, "cluster_weight") and m.cluster_weight is not None:
            init_weight(m.cluster_weight)
        if hasattr(m, "cluster_bias") and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, "out_projs"):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    init_embed(m.out_projs[i])
    elif classname.find("LayerNorm") != -1:
        if hasattr(m, "weight"):
            nn.init.normal_(m.weight, 1.0, cfg.INITIALIZER.base_init[1])
        if hasattr(m, "bias") and m.bias is not None:
            init_bias(m.bias)
    elif classname.find("TransformerLM") != -1:
        if hasattr(m, "r_emb"):
            init_weight(m.r_emb)
        if hasattr(m, "r_w_bias"):
            init_weight(m.r_w_bias)
        if hasattr(m, "r_r_bias"):
            init_weight(m.r_r_bias)
        if hasattr(m, "r_bias"):
            init_bias(m.r_bias)


def update_dropout(m):
    classname = m.__class__.__name__
    if classname.find("Dropout") != -1:
        if hasattr(m, "p"):
            m.p = cfg.MODEL.dropout


def update_dropatt(m):
    if hasattr(m, "dropatt"):
        m.dropatt.p = cfg.MODEL.attention_dropout


assert cfg.MODEL.units % cfg.MODEL.num_heads == 0
model = TransformerGAN(cfg, vocab)
if not args.restart:
    model.generator.apply(weights_init)
    model.generator.word_emb.apply(
        weights_init
    )  # ensure embedding init is not overridden by out_layer in case of weight sharing

args.n_all_param = sum([p.nelement() for p in model.parameters()])
args.n_nonemb_param_gen = sum(
    [p.nelement() for p in model.generator.layers.parameters()]
)
if cfg.DISCRIMINATOR.type != "Null" and cfg.DISCRIMINATOR.type != "":
    args.n_nonemb_param_dis = sum(
        [p.nelement() for p in model.discriminator.parameters()]
    )

    for p in model.discriminator.parameters():
        p.requires_grad = False

if 'ppo' in cfg.DISCRIMINATOR.BERT.loss_type or 'ppo' in cfg.DISCRIMINATOR.CNN.loss_type:
    for p in model.dis_D.parameters():
        p.requires_grad = False

model = model.to(device)

# MLE optimizer
local_lr = cfg.TRAIN.lr / num_gpus
if cfg.TRAIN.optim.lower() == "adam":
    optimizer = optim.Adam(model.generator.parameters(), lr=local_lr,
                           weight_decay=cfg.TRAIN.weight_decay)
elif cfg.TRAIN.optim.lower() == "lamb":
    optimizer = lamb.Lamb(model.generator.parameters(), lr=local_lr,
                          weight_decay=cfg.TRAIN.weight_decay)
elif cfg.TRAIN.optim.lower() == "jitlamb":
    optimizer = lamb.JITLamb(model.generator.parameters(), lr=local_lr,
                             weight_decay=cfg.TRAIN.weight_decay)
else:
    # TODO Add more options
    raise NotImplementedError

# Gen optimizer TODO: URGENT
gen_optimizer = None
if cfg.DISCRIMINATOR.type != "Null" and cfg.DISCRIMINATOR.type != "":
    local_lr = cfg.DISCRIMINATOR.gen_lr / num_gpus
    gen_optimizer = optim.Adam(model.generator.parameters(), lr=local_lr)

# Discriminator optimizer
dis_optimizer = None
if not cfg.DISCRIMINATOR.freeze_discriminator:
    if cfg.DISCRIMINATOR.type == "bert":
        no_decay = ["bias", "LayerNorm.weight"]
        dis_optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.discriminator.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": cfg.DISCRIMINATOR.BERT.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.discriminator.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        dis_optimizer = AdamW(
            dis_optimizer_grouped_parameters,
            lr=cfg.DISCRIMINATOR.BERT.learning_rate,
            eps=cfg.DISCRIMINATOR.BERT.adam_epsilon,
        )

    elif cfg.DISCRIMINATOR.type == "cnn":
        dis_optimizer = optim.Adam(
            model.discriminator.parameters(), lr=cfg.DISCRIMINATOR.CNN.learning_rate
        )
    else:
        pass

if 'ppo' in cfg.DISCRIMINATOR.BERT.loss_type or 'ppo' in cfg.DISCRIMINATOR.CNN.loss_type:
    dis_D_optimizer = optim.Adam(model.dis_D.parameters(), lr=cfg.PPO.dis_D_lr)

if args.fp16:
    if dis_optimizer:
        model, [optimizer, dis_optimizer] = amp.initialize(
            model, [optimizer, dis_optimizer], opt_level="O1", num_losses=2
        )
    else:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level="O1", num_losses=1
        )

#### scheduler
if cfg.TRAIN.scheduler == "cosine":
    # here we do not set eta_min to lr_min to be backward compatible
    # because in previous versions eta_min is default to 0
    # rather than the default value of lr_min 1e-6
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.TRAIN.max_step, eta_min=cfg.TRAIN.lr_min
    )  # should use eta_min arg
elif cfg.TRAIN.scheduler == "inv_sqrt":
    # originally used for Transformer (in Attention is all you need)
    def lr_lambda(step):
        # return a multiplier instead of a learning rate
        if step == 0 and cfg.TRAIN.warmup_step == 0:
            return 1.0
        else:
            return (
                max(
                    (cfg.TRAIN.warmup_step ** 0.5) / (step ** 0.5),
                    cfg.TRAIN.lr_min / cfg.TRAIN.lr,
                )
                if step > cfg.TRAIN.warmup_step
                else step / cfg.TRAIN.warmup_step
            )


    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
elif cfg.TRAIN.scheduler == "dev_perf":
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=cfg.TRAIN.decay_rate,
        patience=cfg.TRAIN.patience,
        min_lr=cfg.TRAIN.lr_min,
    )
elif cfg.TRAIN.scheduler == "constant":
    pass

##Gen scheduler
gen_scheduler = None
if cfg.DISCRIMINATOR.gen_scheduler == "cosine":
    # here we do not set eta_min to lr_min to be backward compatible
    # because in previous versions eta_min is default to 0
    # rather than the default value of lr_min 1e-6
    gen_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        gen_optimizer, cfg.TRAIN.max_step, eta_min=cfg.DISCRIMINATOR.gen_lr_min
    )  # should use eta_min arg
elif cfg.DISCRIMINATOR.gen_scheduler == "inv_sqrt":
    # originally used for Transformer (in Attention is all you need)
    def lr_lambda(step):
        # return a multiplier instead of a learning rate
        if step == 0 and cfg.DISCRIMINATOR.gen_warmup_step == 0:
            return 1.0
        else:
            return (
                max(
                    (cfg.DISCRIMINATOR.gen_warmup_step ** 0.5) / (step ** 0.5),
                    cfg.DISCRIMINATOR.gen_lr_min / cfg.DISCRIMINATOR.gen_lr,
                )
                if step > cfg.DISCRIMINATOR.gen_warmup_step
                else step / cfg.DISCRIMINATOR.gen_warmup_step
            )


    gen_scheduler = optim.lr_scheduler.LambdaLR(gen_optimizer, lr_lambda=lr_lambda)
elif cfg.DISCRIMINATOR.gen_scheduler == "dev_perf":
    gen_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        gen_optimizer,
        factor=cfg.DISCRIMINATOR.gen_decay_rate,
        patience=cfg.DISCRIMINATOR.gen_patience,
        min_lr=cfg.DISCRIMINATOR.gen_lr_min,
    )
elif cfg.DISCRIMINATOR.gen_scheduler == "constant":
    pass

dis_scheduler = None
if cfg.DISCRIMINATOR.dis_scheduler == "cosine":
    # here we do not set eta_min to lr_min to be backward compatible
    # because in previous versions eta_min is default to 0
    # rather than the default value of lr_min 1e-6
    dis_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        dis_optimizer, cfg.TRAIN.max_step, eta_min=cfg.DISCRIMINATOR.dis_lr_min
    )  # should use eta_min arg
elif cfg.DISCRIMINATOR.dis_scheduler == "inv_sqrt":
    # originally used for Transformer (in Attention is all you need)
    def lr_lambda(step):
        # return a multiplier instead of a learning rate
        if step == 0 and cfg.DISCRIMINATOR.dis_warmup_step == 0:
            return 1.0
        else:
            return (
                max(
                    (cfg.DISCRIMINATOR.dis_warmup_step ** 0.5) / (step ** 0.5),
                    cfg.DISCRIMINATOR.dis_lr_min / cfg.DISCRIMINATOR.dis_lr,
                )
                if step > cfg.DISCRIMINATOR.dis_warmup_step
                else step / cfg.DISCRIMINATOR.dis_warmup_step
            )


    dis_scheduler = optim.lr_scheduler.LambdaLR(dis_optimizer, lr_lambda=lr_lambda)
elif cfg.DISCRIMINATOR.dis_scheduler == "dev_perf":
    dis_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        dis_optimizer,
        factor=cfg.DISCRIMINATOR.dis_decay_rate,
        patience=cfg.DISCRIMINATOR.dis_patience,
        min_lr=cfg.DISCRIMINATOR.dis_lr_min,
    )
elif cfg.DISCRIMINATOR.dis_scheduler == "constant":
    pass

# Uncomment if we choose to save DDP model checkpoint
# if args.restart:
#     logging.info('Restarting from checkpoint')

#     map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
#     checkpoint = torch.load(os.path.join(args.work_dir,'checkpoint.pt'))

#     model.load_state_dict(checkpoint['model'],map_location=map_location)
#     optimizer.load_state_dict(checkpoint['optimizer'],map_location=map_location)
#     if dis_optimizer :
#         dis_optimizer.load_state_dict(checkpoint['dis_optimizer'],map_location=map_location)
#     if args.fp16 :
#         amp.load_state_dict(checkpoint['amp'],map_location=map_location)

#     # with open(os.path.join(args.restart_dir, 'checkpoint.pt'), 'rb') as f:
#     #     model = torch.load(f)
#     # if not args.fp16:
#     #     model = model.float()
#     model.apply(update_dropout)
#     model.apply(update_dropatt)
#     dist.barrier()
if cfg.TRAIN.load_from_previous != "Null" and cfg.TRAIN.load_from_previous != "":
    logging.info("Restarting from best model")

    map_location = {"cuda:%d" % 0: "cuda:%d" % args.local_rank}
    checkpoint = torch.load(cfg.TRAIN.load_from_previous, map_location=map_location)

    trimmed_checkpoint = {}
    for key, val in checkpoint["model"].items():
        if "generator" in key:
            new_key = key.replace("generator.", "")
            trimmed_checkpoint[new_key] = val

    model.generator.load_state_dict(trimmed_checkpoint, strict=False)

    # optimizer.load_state_dict(checkpoint["optimizer"])

    model.apply(update_dropout)
    model.apply(update_dropatt)
    dist.barrier()

train_step = 0
best_val_nll = np.inf
# Choose to save non DDP (module) checkpoint
if args.restart:
    # TODO: reload epoch and scheduler

    dst = f"cuda:{args.local_rank}"
    path = os.path.join(args.work_dir, "checkpoint_last.pt")
    logging.info(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=dst)

    train_step = checkpoint["train_step"]
    best_val_nll = checkpoint["best_val_loss"]

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    if dis_optimizer:
        dis_optimizer.load_state_dict(checkpoint["dis_optimizer"])
    if dis_scheduler:
        dis_scheduler.load_state_dict(checkpoint["dis_scheduler"])
    if gen_optimizer:
        gen_optimizer.load_state_dict(checkpoint["gen_optimizer"])
    if gen_scheduler:
        gen_scheduler.load_state_dict(checkpoint["gen_scheduler"])

    if args.fp16:
        amp.load_state_dict(checkpoint["amp"])

    # with open(os.path.join(args.restart_dir, 'checkpoint.pt'), 'rb') as f:
    #     model = torch.load(f)
    # if not args.fp16:
    #     model = model.float()
    model.apply(update_dropout)
    model.apply(update_dropatt)

# model = DDP(model, delay_allreduce=True)
model = DDP(
    model,
    device_ids=[args.local_rank],
    output_device=args.local_rank,
    broadcast_buffers=False,
    find_unused_parameters=False,
)
# model = DDP(model, find_unused_parameters=True)


# if args.restart:
#     if os.path.exists(os.path.join(args.restart_dir, 'optimizer.pt')):
#         with open(os.path.join(args.restart_dir, 'optimizer.pt'), 'rb') as f:
#             opt_state_dict = torch.load(f)
#             optimizer.load_state_dict(opt_state_dict)
#     else:
#         print('Optimizer was not saved. Start from scratch.')

logging.info("=" * 100)
logging.info(args)
logging.info("=" * 100)
logging.info("#total params = {}".format(args.n_all_param))
logging.info("#non emb params in generator = {}".format(args.n_nonemb_param_gen))
if cfg.DISCRIMINATOR.type != "Null" and cfg.DISCRIMINATOR.type != "":
    logging.info("#params in discriminator = {}".format(args.n_nonemb_param_dis))

###############################################################################
# Training code
###############################################################################
print("Start training")


def tensor_to_tokens(tensor):
    """transform Tensor to word tokens"""
    tokens = []
    for sent in tensor:
        sent_token = []
        for word in sent.tolist():
            # if word == cfg.padding_idx:
            #     break
            sent_token.append(word)
        tokens.append(sent_token)
    return tokens


def generate_tokens(num_samples, temperature, batch_size=128, seq_len=2048):
    # Currently only generates unconditional (assuming src is None)

    # First token has to be expanded into one-hot if data is in index form
    assert num_samples % batch_size == 0

    num_samples = num_samples // batch_size
    result = torch.tensor([], dtype=torch.long).to(device)

    cache_tgt_len, cache_mem_len = (
        model.module.generator.tgt_len,
        model.module.generator.mem_len,
    )

    # Reset params for sampling
    model.module.generator.reset_length(1, seq_len)  # Use mem_len=bert_len

    for _ in range(num_samples):
        seq = [torch.zeros(batch_size, dtype=torch.long, device=device)[None, :]]
        sample_mems = None
        status_vec = None


        for _ in range(
                seq_len - 1
        ):  # Since start token is chosen and bert tgt len is fixed

            inp = seq[-1]
            if cfg.TRAIN.append_note_status:
                bptt, batch_size = inp.shape
                if status_vec is None:
                    status_vec = inp.new_zeros((bptt, batch_size, vocab.vec_len), dtype=torch.bool)
                else:
                    status_vec = status_vec[-1:, :, :]
                vocab.update_status_vec(inp, status_vec)

            ret = model.module.generator.forward_generate_gumbel(
                inp, temperature, sample_mems, status_vec=status_vec
            )

            logits, sample_mems = ret
            seq.append(torch.argmax(logits[0], dim=-1)[None, :])

        result = torch.cat([result, torch.cat(seq, dim=0)], dim=1)

    # Reset params for sampling
    model.module.generator.reset_length(
        cache_tgt_len, cache_mem_len
    )  # Use mem_len=bert_len

    return result


def evaluate(eval_iter, dis_val_iter=None, mode="eval", temperature=1):
    # Turn on evaluation mode def disables dropout.
    model.eval()

    # If the model does not use memory at all, make the ext_len longer.
    # Otherwise, make the mem_len longer and keep the ext_len the same.
    if isinstance(model, DDP):
        eval_model = model.module
    else:
        eval_model = model

    eval_model.generator.reset_length(
        tgt_len=cfg.EVALUATE.tgt_length, mem_len=cfg.EVALUATE.mem_length)
    eval_model.generator.same_length = True
    # Evaluation
    total_token_num = 0
    total_nll = 0.0
    # total_gen_len, total_gen_loss = 0, 0

    with torch.no_grad():
        mems = None

        for i, (data, target, all_reset_mem, batch_token_num, status_vec) in enumerate(eval_iter()):

            if all_reset_mem:
                mems = None

            ret = model(data, target, None, "mle", mems, status_vec=status_vec)
            loss, mems = ret["mle"], ret["mems"]
            loss = loss[target != dataset.vocab.pad_id]
            loss = loss.mean()
            total_nll += batch_token_num * loss.float().item()
            total_token_num += batch_token_num

        # Compute metrics (what size ref corpus do we consider)
        gen_tokens = None
        if cfg.METRICS.use_bleu:
            gen_tokens = tensor_to_tokens(
                generate_tokens(625, temperature).transpose(0, 1)
            )

            if mode == "eval":
                all_text = [el.tolist() for el in dataset._valid_data]
            else:
                all_text = [el.tolist() for el in dataset._test_data]
            bleu.reset(test_text=gen_tokens, real_text=all_text)

        if cfg.METRICS.use_self_bleu and mode == "eval":
            if not cfg.METRICS.use_bleu:
                gen_tokens = tensor_to_tokens(
                    generate_tokens(625, temperature).transpose(0, 1)
                )

            gen_tokens_s = tensor_to_tokens(
                generate_tokens(2500, temperature).transpose(0, 1)
            )
            self_bleu.reset(test_text=gen_tokens_s, real_text=gen_tokens)

        if cfg.METRICS.CLASSIFIER.use_classifier and mode == "eval":
            gen_tokens = generate_tokens(cfg.METRICS.CLASSIFIER.gen_num_samples, temperature,
                                         batch_size=cfg.METRICS.CLASSIFIER.gen_batch_size,
                                         seq_len=cfg.METRICS.CLASSIFIER.gen_seq_len).transpose(0, 1)
            if mode == "eval":
                all_text = [el for el in dataset._valid_data]

            classifier.reset(test_text=gen_tokens, real_text=all_text)

        if mode == "eval":
            results = [metric.get_score() for metric in eval_metrics]
            if gen_tokens is not None:
                del gen_tokens
        else:
            results = [metric.get_score() for metric in test_metrics]
            # Switch back to the training mode


    eval_model.generator.reset_length(cfg.TRAIN.tgt_length, cfg.TRAIN.mem_length)
    eval_model.generator.same_length = cfg.MODEL.same_length

    model.train()

    return total_token_num, total_nll, results  # , total_gen_loss, total_gen_len


def train():
    global train_step
    global best_val_nll

    log_train_loss = torch.tensor(0.0).float().to(device)
    log_grad_norm = torch.tensor(0.0).float().to(device)
    log_token_num = torch.tensor(0).to(device)

    # Discriminator related
    log_gen_train_loss = torch.tensor(0.0).float().to(device)  # Log discriminator loss
    log_gen_num = torch.tensor(0.0).float().to(device)

    log_dis_train_loss = torch.tensor(0.0).float().to(device)
    log_dis_num = torch.tensor(0.0).float().to(device)

    dis_iterations = 0  # Num dis iters
    best_gen_val_loss = np.inf

    if cfg.DISCRIMINATOR.type != "Null" and cfg.DISCRIMINATOR.type != "":
        dis_iterator = dis_iter()

    log_start_time = time.time()  # coding: utf-8

    mems = [None for _ in range(cfg.TRAIN.batch_chunk)]

    assert batch_size % cfg.TRAIN.batch_chunk == 0
    train_real_iter = train_iter()

    for batch, (data, target, reset_mems, batch_token_num, status_vec) in enumerate(
            train_real_iter
    ):
        beta = get_fixed_temperature(
            cfg.DISCRIMINATOR.beta_max,
            train_step,
            cfg.TRAIN.max_step,
            cfg.DISCRIMINATOR.adapt,
        )
        model.module.temperature = 1.0 / beta

        model.zero_grad()

        # Batch chunking

        data_chunks = torch.chunk(data, cfg.TRAIN.batch_chunk, 1)
        target_chunks = torch.chunk(target, cfg.TRAIN.batch_chunk, 1)
        reset_mems_chunks = torch.chunk(reset_mems, cfg.TRAIN.batch_chunk, 0)
        if status_vec is not None:
            status_vec_chunks = torch.chunk(status_vec, cfg.TRAIN.batch_chunk, 1)
        for i in range(cfg.TRAIN.batch_chunk):

            data = data_chunks[i].contiguous()
            target = target_chunks[i].contiguous()
            reset_mems = reset_mems_chunks[i].contiguous()
            if status_vec is not None:
                status_vec = status_vec_chunks[i].contiguous()

            # reset_mems = None
            ret = model(data, target, reset_mems, "mle", mems[i], status_vec=status_vec)
            loss, mems[i] = ret["mle"], ret["mems"]

            loss = loss[target != dataset.vocab.pad_id]
            loss = loss.float().mean() / cfg.TRAIN.batch_chunk
            log_train_loss += (
                    loss.item()
                    * (target != dataset.vocab.pad_id).sum()
                    * cfg.TRAIN.batch_chunk
            )

            if cfg.TRAIN.use_mle:
                if args.fp16:
                    with amp.scale_loss(loss, optimizer, loss_id=1) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

        log_token_num += int(batch_token_num)

        if cfg.TRAIN.use_mle:
            if args.fp16:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), cfg.TRAIN.clip
                )
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.module.generator.parameters(), cfg.TRAIN.clip
                )

            # a = [torch.norm(w.grad) for w in model.module.generator.parameters()]
            log_grad_norm += grad_norm
            optimizer.step()
            optimizer.zero_grad()

        # Train discriminator
        if train_step > cfg.DISCRIMINATOR.start_iter and (
                train_step % cfg.DISCRIMINATOR.dis_loss_freq == 0
        ):
            # TODO: dis training messes up memory structure maintained during batch loading
            # (we need another dataloader foor real data)
            if not (cfg.DISCRIMINATOR.freeze_discriminator):
                for dis_iterations in range(cfg.DISCRIMINATOR.dis_steps):

                    try:
                        dis_data, _ = next(dis_iterator)
                    except StopIteration:
                        dis_iterator = dis_iter()

                    # Batch chunking for generator and discriminator
                    dis_data_chunks = torch.chunk(
                        dis_data, cfg.DISCRIMINATOR.batch_chunk, 1
                    )

                    if cfg.DISCRIMINATOR.type == "bert":
                        for idx, p in enumerate(
                                model.module.discriminator.parameters()
                        ):
                            if idx in model.module.discriminator.unfreeze_idx:
                                p.requires_grad = True
                    else:
                        for p in model.module.discriminator.parameters():
                            p.requires_grad = True

                    for i in range(cfg.DISCRIMINATOR.batch_chunk):
                        dis_data = dis_data_chunks[i].contiguous()
                        # Share the same mems with mle iter
                        ret = model(dis_data, None, None, "dis_loss")

                        dis_loss = ret["dis_loss"]
                        log_dis_train_loss += dis_loss.float().item()
                        dis_loss = (
                                dis_loss.float().mean() / cfg.DISCRIMINATOR.batch_chunk
                        )
                        if (
                                cfg.DISCRIMINATOR.type == "bert"
                                and "gp" in cfg.DISCRIMINATOR.BERT.loss_type
                        ):
                            gp_loss = ret["gp_loss"]
                            gp_loss = (
                                    gp_loss.float().mean() / cfg.DISCRIMINATOR.batch_chunk
                            )
                        elif (
                                cfg.DISCRIMINATOR.type == "cnn"
                                and "gp" in cfg.DISCRIMINATOR.CNN.loss_type
                        ):
                            gp_loss = ret["gp_loss"]
                            gp_loss = (
                                    gp_loss.float().mean() / cfg.DISCRIMINATOR.batch_chunk
                            )

                        log_dis_num += 1

                        if args.fp16:
                            with amp.scale_loss(
                                    dis_loss, dis_optimizer, loss_id=0
                            ) as scaled_dis_loss:
                                scaled_dis_loss.backward()
                        else:
                            if not cfg.DISCRIMINATOR.backprop_outside:
                                dis_loss.backward()
                                if (
                                        cfg.DISCRIMINATOR.type == "bert"
                                        and "gp" in cfg.DISCRIMINATOR.BERT.loss_type
                                ):
                                    gp_loss.backward()
                                elif (
                                        cfg.DISCRIMINATOR.type == "cnn"
                                        and "gp" in cfg.DISCRIMINATOR.CNN.loss_type
                                ):
                                    gp_loss.backward()

                    # TODO: investigate training tricks for dis different clip?
                    if args.fp16:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            amp.master_params(dis_optimizer), cfg.TRAIN.clip
                        )
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.module.discriminator.parameters(), cfg.TRAIN.clip,
                        )

                    dis_optimizer.step()
                    dis_optimizer.zero_grad()

            for p in model.module.discriminator.parameters():
                p.requires_grad = False

        if train_step > cfg.DISCRIMINATOR.start_iter and (
                train_step % cfg.DISCRIMINATOR.gen_loss_freq == 0
        ):

            # Train generator
            # Make dis parameters non trainable
            try:
                dis_data, _ = next(dis_iterator)
            except StopIteration:
                dis_iterator = dis_iter()

            # Batch chunking for generator and discriminator
            dis_data_chunks = torch.chunk(dis_data, cfg.DISCRIMINATOR.batch_chunk, 1)

            for i in range(cfg.DISCRIMINATOR.batch_chunk):
                dis_data = dis_data_chunks[i].contiguous()

                update_D0 = False
                if train_step % cfg.PPO.dis_D_update_D0_freq == 0:
                    update_D0 = True

                if 'ppo' in cfg.DISCRIMINATOR.BERT.loss_type or 'ppo' in cfg.DISCRIMINATOR.CNN.loss_type:
                    for p in model.module.dis_D.parameters():
                        p.requires_grad = True

                    #Use same real batch and generate new fake batch
                    # Always backprop outside
                    ret = model(dis_data, None, None, "classifier_loss")
                    torch.nn.utils.clip_grad_norm_(model.module.dis_D.parameters(), cfg.TRAIN.clip)
                    dis_D_optimizer.step()
                    dis_D_optimizer.zero_grad()

                    for p in model.module.dis_D.parameters():
                        p.requires_grad = False


                ret = model(dis_data, None, None, "gen_loss", update_D0=update_D0)

                gen_loss = ret["gen_loss"]
                log_gen_train_loss += gen_loss.float().item()

                gen_loss = gen_loss.float().mean() / cfg.DISCRIMINATOR.batch_chunk
                log_gen_num += 1

                # if args.fp16:
                #     with amp.scale_loss(gen_loss, optimizer, loss_id=2) as scaled_gen_loss:
                #         scaled_gen_loss.backward(retain_graph=True)
                # else:
                #     gen_loss.backward(retain_graph=True)

                if args.fp16:
                    with amp.scale_loss(gen_loss, optimizer, loss_id=1) as scaled_loss:
                        scaled_loss.backward()
                else:
                    # a = [torch.norm(w.grad) for w in model.module.generator.parameters()]
                    if not cfg.DISCRIMINATOR.backprop_outside:
                        gen_loss.backward()
                    # b = [torch.norm(w.grad) for w in model.module.generator.parameters()]
                    # c = [(j-i) for i,j in zip(a,b)]
                    # d = ([i/j for i,j in zip(a,c)])
                    # d = sum(d)/len(d)

                    pass

            if args.fp16:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), cfg.TRAIN.clip
                )
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.module.generator.parameters(), cfg.TRAIN.clip
                )

            gen_optimizer.step()
            gen_optimizer.zero_grad()



        # step-wise learning rate annealing
        train_step += 1

        if cfg.TRAIN.scheduler in ["cosine", "constant", "dev_perf"]:
            # linear warmup stage
            if train_step < cfg.TRAIN.warmup_step:
                curr_lr = cfg.TRAIN.lr * train_step / cfg.TRAIN.warmup_step
                optimizer.param_groups[0]["lr"] = curr_lr
            else:
                if cfg.TRAIN.scheduler == "cosine":
                    scheduler.step()
        elif cfg.TRAIN.scheduler == "inv_sqrt":
            scheduler.step()

        if cfg.DISCRIMINATOR.type != "Null" and cfg.DISCRIMINATOR.type != "":
            if cfg.DISCRIMINATOR.gen_scheduler in ["cosine", "constant", "dev_perf"]:
                # linear warmup stage
                if train_step < cfg.DISCRIMINATOR.gen_warmup_step:
                    curr_gen_lr = (
                            cfg.DISCRIMINATOR.gen_lr * train_step / cfg.TRAIN.warmup_step
                    )
                    gen_optimizer.param_groups[0]["lr"] = curr_gen_lr
                else:
                    if cfg.DISCRIMINATOR.gen_scheduler == "cosine":
                        gen_scheduler.step()
            elif cfg.DISCRIMINATOR.gen_scheduler == "inv_sqrt":
                gen_scheduler.step()

            if cfg.DISCRIMINATOR.dis_scheduler in ["cosine", "constant", "dev_perf"]:
                # linear warmup stage
                if train_step < cfg.DISCRIMINATOR.dis_warmup_step:
                    curr_dis_lr = (
                            cfg.DISCRIMINATOR.dis_lr * train_step / cfg.TRAIN.warmup_step
                    )
                    dis_optimizer.param_groups[0]["lr"] = curr_dis_lr
                else:
                    if cfg.DISCRIMINATOR.dis_scheduler == "cosine":
                        dis_scheduler.step()
            elif cfg.DISCRIMINATOR.dis_scheduler == "inv_sqrt":
                dis_scheduler.step()

        if train_step % cfg.TRAIN.log_interval == 0:
            torch.distributed.all_reduce(log_train_loss)
            torch.distributed.all_reduce(log_grad_norm)
            torch.distributed.all_reduce(log_token_num)

            torch.distributed.all_reduce(log_gen_train_loss)
            torch.distributed.all_reduce(log_gen_num)

            log_train_loss /= log_token_num
            log_grad_norm /= cfg.TRAIN.log_interval * num_gpus
            log_gen_train_loss = (
                log_gen_train_loss / log_gen_num
                if log_gen_num != 0
                else torch.tensor(0.0).float().to(device)
            )
            log_dis_train_loss = (
                log_dis_train_loss / log_dis_num
                if log_dis_num != 0
                else torch.tensor(0.0).float().to(device)
            )
            if args.local_rank == 0:
                elapsed = time.time() - log_start_time
                logging.info(
                    "Train Step {}/{}, lr={:f}, tokens/s={:.1f},"
                    " nll={:.4f}, ppl={:.2f}, grad norm={}, gen_loss={:5.4f}, dis_loss={:5.4f}".format(
                        train_step,
                        cfg.TRAIN.max_step,
                        optimizer.param_groups[0]["lr"],
                        log_token_num.item() / elapsed,
                        log_train_loss.item(),
                        math.exp(log_train_loss.item()),
                        log_grad_norm.item(),
                        log_gen_train_loss.item(),
                        log_dis_train_loss.item(),
                    )
                )

            log_train_loss[()] = 0
            log_grad_norm[()] = 0
            log_token_num[()] = 0

            log_gen_train_loss[()] = 0
            log_gen_num[()] = 0

            log_dis_train_loss[()] = 0
            log_dis_num[()] = 0

            log_start_time = time.time()

        if train_step % cfg.TRAIN.eval_interval == 0:
            eval_start_time = time.time()

            val_token_num, val_total_nll, val_metrics = evaluate(
                eval_iter=val_iter, dis_val_iter=None, mode="eval"
            )

            val_token_num_pt = torch.tensor(val_token_num).to(device)
            val_total_nll_pt = torch.tensor(val_total_nll / 10000.0).to(device)

            torch.distributed.all_reduce(val_token_num_pt)
            torch.distributed.all_reduce(val_total_nll_pt)

            val_token_num = val_token_num_pt.item()
            val_total_nll = val_total_nll_pt.item()

            val_nll = val_total_nll / (val_token_num / 10000.0)

            if args.local_rank == 0:
                logging.info(
                    "Eval step {}, time={}s, val nll={}, val ppl={}, #evaluated tokens={}, bleu={}, self_bleu={"
                    "}, class_acc={}".format(
                        train_step,
                        time.time() - eval_start_time,
                        val_nll,
                        math.exp(val_nll),
                        val_token_num,
                        val_metrics[0],
                        val_metrics[1],
                        val_metrics[2],
                    )
                )
            # Save the model if the validation loss is the best we've seen so far.

            # Always save after eval if save_all is true and not debug
            if not args.debug and args.save_all:
                name = f"checkpoint_{train_step}.pt"
                save_checkpoint(
                    args,
                    model,
                    optimizer,
                    dis_optimizer,
                    gen_optimizer,
                    dataset.vocab,
                    train_step,
                    val_nll,
                    scheduler,
                    dis_scheduler,
                    gen_scheduler,
                    name,
                )

            # Save last checkpoint if not debug and not save_all
            if not args.debug and not args.save_all:
                name = "checkpoint_last.pt"
                save_checkpoint(
                    args,
                    model,
                    optimizer,
                    dis_optimizer,
                    gen_optimizer,
                    dataset.vocab,
                    train_step,
                    val_nll,
                    scheduler,
                    dis_scheduler,
                    gen_scheduler,
                    name,
                )

            if not best_val_nll or val_nll < best_val_nll:
                best_val_nll = val_nll

                if not args.debug:
                    name = "checkpoint_best.pt"
                    save_checkpoint(
                        args,
                        model,
                        optimizer,
                        dis_optimizer,
                        gen_optimizer,
                        dataset.vocab,
                        train_step,
                        best_val_nll,
                        scheduler,
                        dis_scheduler,
                        gen_scheduler,
                        name,
                    )

                test_start_time = time.time()

                def calculate_test_nll_during_training(test_iter):

                    # Run on test data.
                    # test_token_num, test_total_nll, test_gen_loss, test_gen_num = evaluate(
                    #     eval_iter=test_iter, dis_val_iter=dis_test_iter
                    # )
                    test_token_num, test_total_nll, test_metrics = evaluate(
                        eval_iter=test_iter, dis_val_iter=None, mode="test"
                    )
                    test_token_num_pt = torch.tensor(test_token_num).to(device)
                    test_total_nll_pt = torch.tensor(test_total_nll / 10000.0).to(
                        device
                    )
                    # test_gen_loss_pt = torch.tensor(test_gen_loss).to(device)
                    # test_gen_num_pt = torch.tensor(test_gen_num).to(device)

                    torch.distributed.all_reduce(test_token_num_pt)
                    torch.distributed.all_reduce(test_total_nll_pt)
                    # torch.distributed.all_reduce(test_gen_loss_pt)
                    # torch.distributed.all_reduce(test_gen_num_pt)

                    test_token_num = test_token_num_pt.item()
                    test_nll = test_total_nll_pt.item() / (test_token_num / 10000.0)

                    # test_gen_loss = test_gen_loss_pt.item()
                    # test_gen_num = test_gen_num_pt.item()
                    # test_gen_loss = (
                    #     test_gen_loss / test_gen_num
                    #     if test_gen_num != 0
                    #     else torch.tensor(0.0).float().to(device)
                    # )
                    return test_token_num, test_nll, test_metrics

                (
                    test_token_num,
                    test_nll,
                    test_metrics,
                ) = calculate_test_nll_during_training(test_iter)

                if args.local_rank == 0:
                    logging.info(
                        "Test step {}, time={}s, test nll={}, test ppl={}, #evaluated tokens={}"
                        " test_bleu={}".format(
                            train_step,
                            time.time() - test_start_time,
                            test_nll,
                            math.exp(test_nll),
                            test_token_num,
                            test_metrics[0],
                        )
                    )
            # dev-performance based learning rate annealing
            if cfg.TRAIN.scheduler == "dev_perf":
                scheduler.step(val_nll)

        if train_step == cfg.TRAIN.max_step:
            logging.info("-" * 100)
            logging.info("End of training")
            break


if __name__ == "__main__":
    train()
    # Load the best saved model.
    cfg.defrost()
    cfg.DISCRIMINATOR.type = "Null"
    cfg.MODEL.same_length = True
    cfg.freeze()
    model = TransformerGAN(cfg, dataset._vocab)
    checkpoint = torch.load(os.path.join(args.work_dir, "checkpoint_best.pt"))

    trimmed_checkpoint = {}
    for key, val in checkpoint["model"].items():
        if "generator" in key:
            new_key = key.replace("generator.", "")
            trimmed_checkpoint[new_key] = val
    model.generator.load_state_dict(trimmed_checkpoint)
    # Do the evaluation of the best model
    model = model.to(device)

    test_token_num, test_total_nll, test_metrics = evaluate(
        eval_iter=test_iter, dis_val_iter=None, mode="test"
    )
    test_token_num_pt = torch.tensor(test_token_num).to(device)
    test_total_nll_pt = torch.tensor(test_total_nll / 10000.0).to(device)
    torch.distributed.all_reduce(test_token_num_pt)
    torch.distributed.all_reduce(test_total_nll_pt)
    test_token_num = test_token_num_pt.item()
    test_nll = test_total_nll_pt.item() / (test_token_num / 10000.0)
    logging.info("=" * 100)
    logging.info(
        "| End of training | test nll {:5.2f} | test ppl {:9.3f}".format(
            test_nll, math.exp(test_nll)
        )
    )
    logging.info("=" * 100)
