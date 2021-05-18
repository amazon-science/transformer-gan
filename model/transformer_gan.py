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

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("utils")
from helpers import get_losses

from transformers import (
    BertConfig,
    BertForMaskedLM,
    PreTrainedTokenizer,
    PreTrainedModel,
    AdamW,
    BertForSequenceClassification
)

from mem_transformer import MemTransformerLM

# Define RelGAN discriminator
from discriminator import CNNDiscriminator

dis_filter_sizes = [2, 3, 4, 5]
dis_num_filters = [300, 300, 300, 300]

# dis_filter_sizes = [5]
# dis_num_filters = [300]


class RelGAN_D(CNNDiscriminator):
    def __init__(
            self,
            embed_dim,
            max_seq_len,
            num_rep,
            vocab_size,
            padding_idx,
            gpu=True,
            dropout=0.25,
            cfg=None,
    ):
        super(RelGAN_D, self).__init__(
            embed_dim,
            vocab_size,
            dis_filter_sizes,
            dis_num_filters,
            padding_idx,
            gpu,
            dropout,
            cfg,
        )

        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.feature_dim = sum(dis_num_filters)
        self.emb_dim_single = int(embed_dim / num_rep)

        self.embeddings = nn.Linear(vocab_size, embed_dim, bias=False)

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    1, n, (f, self.emb_dim_single), stride=(1, self.emb_dim_single)
                )
                for (n, f) in zip(dis_num_filters, dis_filter_sizes)
            ]
        )

        self.highway = nn.Linear(self.feature_dim, self.feature_dim)
        self.feature2out = nn.Linear(self.feature_dim, 100)
        self.out2logits = nn.Linear(100, 1)
        self.dropout = nn.Dropout(dropout)

        self.init_params()

    def forward(self, inp):
        """
        Get logits of discriminator
        :param inp: batch_size * seq_len * vocab_size
        :return logits: [batch_size * num_rep] (1-D tensor)
        """
        emb = self.embeddings(inp).unsqueeze(
            1
        )  # batch_size * 1 * max_seq_len * embed_dim

        cons = [
            F.relu(conv(emb)) for conv in self.convs
        ]  # [batch_size * num_filter * (seq_len-k_h+1) * num_rep]
        pools = [
            F.max_pool2d(con, (con.size(2), 1)).squeeze(2) for con in cons
        ]  # [batch_size * num_filter * num_rep]
        pred = torch.cat(pools, 1)
        pred = (
            pred.permute(0, 2, 1).contiguous().view(-1, self.feature_dim)
        )  # (batch_size * num_rep) * feature_dim
        highway = self.highway(pred)
        pred = (
                torch.sigmoid(highway) * F.relu(highway)
                + (1.0 - torch.sigmoid(highway)) * pred
        )  # highway

        pred = self.feature2out(self.dropout(pred))
        logits = self.out2logits(pred).squeeze(1)  # [batch_size * num_rep]

        return logits


class TransformerGAN(nn.Module):
    def __init__(self, cfg, vocab):

        super(TransformerGAN, self).__init__()

        self.ntokens = len(vocab)
        self.generator = MemTransformerLM(
            cfg,
            self.ntokens,
            vocab.vec_len,
        )

        # select_discriminator = args.select_discriminator, dis_cfg = dis_cfg
        def create_dis_D():
            if cfg.PPO.dis_D_type == "bert":
                dis_D = self.create_bert_model(
                    cfg.DISCRIMINATOR.BERT.model_path, cfg.DISCRIMINATOR.BERT.loss_type,
                    cfg.DISCRIMINATOR.BERT.model_type
                )
                dis_D.unfreeze_idx = self.calculate_unfreeze_idx(cfg)
            elif cfg.PPO.dis_D_type == "cnn": \
                    dis_D = RelGAN_D(
                        cfg.DISCRIMINATOR.CNN.embed_dim,
                        cfg.DISCRIMINATOR.tgt_len,
                        cfg.PPO.dis_D_num_rep,  # Has to be 1 if used with BERT
                        self.ntokens,
                        1,
                        cfg=cfg,
                    )
            return dis_D

        if 'ppo' in cfg.DISCRIMINATOR.CNN.loss_type or 'ppo' in cfg.DISCRIMINATOR.BERT.loss_type:
            self.dis_D = create_dis_D()
            self.P0 = None

        # Create discriminator
        if cfg.DISCRIMINATOR.type == "bert":
            # Can change d_embed
            self.discriminator = self.create_bert_model(
                cfg.DISCRIMINATOR.BERT.model_path, cfg.DISCRIMINATOR.BERT.loss_type, cfg.DISCRIMINATOR.BERT.model_type,
                cfg.DISCRIMINATOR.BERT.random_weights
            )
            self.discriminator.unfreeze_idx = self.calculate_unfreeze_idx(cfg)

        elif cfg.DISCRIMINATOR.type == "cnn":
            self.discriminator = RelGAN_D(
                cfg.DISCRIMINATOR.CNN.embed_dim,
                cfg.DISCRIMINATOR.tgt_len,
                cfg.DISCRIMINATOR.CNN.num_rep,
                self.ntokens,
                1,
                cfg=cfg,
            )

        else:
            self.discriminator = None

        self.cfg = cfg
        self.temperature = 1
        self.vocab = vocab
        self.vec_len = vocab.vec_len

    def dis_D_forward(self, data):
        data = torch.transpose(data, 0, 1)
        if self.cfg.PPO.dis_D_type == "bert":
            embedding_matrix = self.dis_D.bert.embeddings.word_embeddings.weight

            if data.ndim == 3:
                data = torch.argmax(data, dim=-1)
            emb = embedding_matrix[data]
            outputs = self.dis_D(inputs_embeds=emb)[0][:, 0]

        elif self.cfg.PPO.dis_D_type == "cnn":
            if data.ndim == 2:
                data = (
                    data.new_zeros((*data.shape, self.ntokens), dtype=torch.float32)
                        .scatter_(-1, data[..., None], 1)
                )
            outputs = self.dis_D(data)
        return outputs

    def calc_gradient_penalty(self, real_data, fake_data, LAMBDA=10):
        alpha = torch.rand([real_data.shape[0], 1, 1], device=real_data.device)
        alpha = alpha.expand(real_data.size())

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
        if 'bert' in self.cfg.DISCRIMINATOR.type:
            interpolates = torch.einsum(
                "ve,bcv -> bce",
                self.discriminator.bert.embeddings.word_embeddings.weight,
                interpolates,
            )
            disc_interpolates = self.discriminator(inputs_embeds=interpolates)[0][:, 0]
        elif 'cnn' in self.cfg.DISCRIMINATOR.type:
            disc_interpolates = self.discriminator(interpolates)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(disc_interpolates.size(),
                                                                device=real_data.device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(real_data.shape[0], -1)

        # https://github.com/igul222/improved_wgan_training/blob/master/gan_language.py
        slopes = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        gradient_penalty = ((slopes - 1.) ** 2).mean() * LAMBDA

        return gradient_penalty

    def forward(self, data, target, reset_mems, train_loss, mems=None, status_vec=None,
                update_D0=False):
        # loss type can be "mle","gen_loss", "dis_loss" or "mle_and_gen_loss"

        return_dict = {"mle": None, "gen_loss": None, "dis_loss": None, "mems": None}

        if "mle" in train_loss:
            ret = self.generator(data, target, reset_mems, mems, status_vec=status_vec)
            return_dict["mle"], return_dict["mems"] = ret

        # Sample a sequence
        if "gen" in train_loss or "dis" in train_loss or "classifier" in train_loss:
            # TODO: low priority could potentially make forward_generate a static func?
            # Cache params
            cache_tgt_len, cache_mem_len = (
                self.generator.tgt_len, self.generator.mem_len
            )

            # Reset params for sampling
            self.generator.reset_length(
                1, self.cfg.DISCRIMINATOR.mem_len
            )  # Use mem_len=bert_len

            # Generate samples
            # First token has to be expanded into one-hot if data is in index form

            # sample_mems can be greater than dis mem_len but only for one forward generate pass

            def process_for_sequence(inp):
                if len(inp.shape) == 1:
                    # Use F.one_hot(,self.n_token)
                    return (
                        inp.new_zeros(
                            (*inp.shape, self.ntokens), dtype=torch.float32
                        ).scatter_(-1, inp[..., None], 1)
                    )
                elif len(inp.shape) == 2:
                    return (inp)
                else:
                    raise NotImplementedError

            seq = []
            sample_mems = None
            # TODO: When training gen do not pass only context into dis (since no grads anyway)
            # TODO: do not loop over context

            status_vec = None
            with torch.no_grad():
                # Last token in context is used to feed forward for sequential generation
                if self.cfg.DISCRIMINATOR.context_len > 1:
                    context = data[:self.cfg.DISCRIMINATOR.context_len - 1]

                    if self.cfg.TRAIN.append_note_status:
                        bptt, batch_size = context.shape
                        status_vec = context.new_zeros((bptt, batch_size, self.vec_len), dtype=torch.bool)
                        self.vocab.update_status_vec(context, status_vec)
                    ret = self.generator.forward_generate(context, sample_mems,
                                                          status_vec=status_vec)
                    _, sample_mems  = ret

            sample_len = self.cfg.DISCRIMINATOR.tgt_len // self.cfg.DISCRIMINATOR.sample_chunks_mem

            # Do not detach the gradient w.r.t memory. Choose to manually detach
            self.generator.detach_mems_grad = False
            gen_loss, dis_loss, gp_loss = 0, 0, 0

            # Split real and fake samples according to sample_chunks_mem
            for chunk_start in range(0, self.cfg.DISCRIMINATOR.tgt_len, sample_len):
                chunk_end = min(chunk_start + sample_len, self.cfg.DISCRIMINATOR.tgt_len)
                # TODO: Can we retain sub graph after calling backward?
                for ind in range(chunk_start, chunk_end):

                    if ind < self.cfg.DISCRIMINATOR.context_len:
                        seq.append(process_for_sequence(data[ind]))
                        continue
                    # Since start token is chosen and bert tgt len is fixed
                    elif self.cfg.DISCRIMINATOR.truncate_backprop or (ind == chunk_start) or "classifier" in train_loss:
                        # Noticed I do not gain much memory if dis is frozen since requires_grad=False for dis params
                        # Also do not gain memory is dis is freed (Assuming pytorch already optimizes and shares comp
                        # graphs)
                        # Saved some time in backward()

                        # Stop gradient propagation
                        inp = torch.argmax(seq[-1], dim=-1)[None,
                              :].detach()  # Gumbel max so gradients do not propagate
                        cont = inp
                    else:
                        inp = seq[-1][None, :]
                        cont = torch.argmax(seq[-1], dim=-1)[None, :].detach()

                    if self.cfg.TRAIN.append_note_status:
                        bptt, batch_size = cont.shape
                        if status_vec is None:
                            status_vec = inp.new_zeros((bptt, batch_size, self.vec_len), dtype=torch.bool)
                        else:
                            status_vec = status_vec[-1:, :, :]
                        self.vocab.update_status_vec(cont, status_vec)

                    ret = self.generator.forward_generate_gumbel(inp, self.temperature, sample_mems, status_vec=status_vec)

                    logits, sample_mems = ret

                    seq.append(logits[0])

                # Prepare fake data

                # Ignore first token
                if len(seq) == sample_len + 1:
                    seq = seq[1:]

                fake_chunk = torch.cat(
                    [i[None, :, :] for i in seq], 0
                )  # seq_len, bsz, vocab

                if 'dis' in train_loss:
                    fake_chunk = fake_chunk.detach()

                data_chunk = data[chunk_start:chunk_end]

                if "classifier" in train_loss:
                    if self.P0 is None:
                        with torch.no_grad():
                            D0 = torch.sigmoid(self.dis_D_forward(fake_chunk))
                            self.P0 = (1. - D0) / torch.clamp(D0, min=1e-7)

                    real_label = self.P0.new_full((self.P0.shape[0],), 1.)
                    fake_label = self.P0.new_full((self.P0.shape[0],), 0.)

                    criterion = nn.BCELoss()
                    errDD_real = criterion(torch.sigmoid(self.dis_D_forward(data_chunk)), real_label)
                    errDD_fake = criterion(torch.sigmoid(self.dis_D_forward(fake_chunk.detach())), fake_label)
                    errDD_loss = errDD_real + errDD_fake

                    ((errDD_loss.float().mean()) / (
                                self.cfg.DISCRIMINATOR.batch_chunk * self.cfg.DISCRIMINATOR.sample_chunks_mem)).backward()


                    # Reset params for next chunk
                    sample_mems = sample_mems.detach()
                    seq = [seq[-1]]

                    continue

                # Train classifier
                if 'gen' in train_loss and (
                        'ppo' in self.cfg.DISCRIMINATOR.BERT.loss_type or 'ppo' in self.cfg.DISCRIMINATOR.CNN.loss_type):
                    if self.P0 is None or update_D0:
                        with torch.no_grad():
                            D0 = torch.sigmoid(self.dis_D_forward(fake_chunk))
                            self.P0 = (1. - D0) / torch.clamp(D0, min=1e-7)

                    D1 = torch.sigmoid(self.dis_D_forward(fake_chunk))
                    P1 = (1. - D1)
                    ratio = (P1 / torch.clamp(D1 * self.P0, min=1e-7))

                    ratio_clipped = torch.clamp(ratio, 1.0 - self.cfg.PPO.clip_param, 1.0 +
                                                self.cfg.PPO.clip_param)


                if self.cfg.DISCRIMINATOR.type == "bert": \
                        # bert_vocab_size = 311

                    data_chunk = torch.transpose(data_chunk, 0, 1)
                    fake_chunk = torch.transpose(fake_chunk, 0, 1)

                    # Pad zeros corresponding to MASK token
                    fake_chunk = torch.cat(
                        [fake_chunk, fake_chunk.new_zeros((*fake_chunk.shape[:-1], 1))], -1
                    )


                    embedding_matrix = self.discriminator.bert.embeddings.word_embeddings.weight

                    emb_real = embedding_matrix[data_chunk]
                    emb_fake = torch.einsum(
                        "ve,bcv -> bce",
                        embedding_matrix,
                        fake_chunk,
                    )

                    bert_emb_real = self.discriminator(inputs_embeds=emb_real)
                    bert_emb_fake = self.discriminator(inputs_embeds=emb_fake)

                    # 1 is real and 0 is fake in bce_loss, so take we extract the real index from output vector
                    d_out_real, d_out_fake = bert_emb_real[0][:, 0], bert_emb_fake[0][:, 0]

                    if 'ppo' in self.cfg.DISCRIMINATOR.BERT.loss_type and 'gen' in train_loss:
                        surr1 = ratio * d_out_fake
                        surr2 = ratio_clipped * d_out_fake
                        target = torch.where(d_out_fake > 0, torch.min(surr1, surr2), torch.max(surr1, surr2))
                        temp_gen_loss, temp_dis_loss = get_losses(d_out_real, target,
                                                                  self.cfg.DISCRIMINATOR.BERT.loss_type)
                    else:
                        temp_gen_loss, temp_dis_loss = get_losses(d_out_real, d_out_fake,
                                                                  self.cfg.DISCRIMINATOR.BERT.loss_type)

                    # Regularize discriminator with gradient penalty
                    if "dis" in train_loss and 'gp' in self.cfg.DISCRIMINATOR.BERT.loss_type:
                        data_chunk = (
                            data_chunk.new_zeros((*data_chunk.shape, self.ntokens + 1), dtype=torch.float32)
                                .scatter_(-1, data_chunk[..., None], 1)
                        )
                        temp_gp_loss = self.calc_gradient_penalty(data_chunk, fake_chunk)

                    if self.cfg.DISCRIMINATOR.backprop_outside:
                        gen_loss += temp_gen_loss.detach()
                        dis_loss += temp_dis_loss.detach()
                        if "dis" in train_loss and 'gp' in self.cfg.DISCRIMINATOR.BERT.loss_type:
                            gp_loss += temp_gp_loss.detach()
                    else:
                        gen_loss += temp_gen_loss
                        dis_loss += temp_dis_loss
                        if "dis" in train_loss and 'gp' in self.cfg.DISCRIMINATOR.BERT.loss_type:
                            gp_loss += temp_gp_loss

                elif self.cfg.DISCRIMINATOR.type == "cnn":
                    real_samples = (
                        data_chunk.new_zeros((*data_chunk.shape, self.ntokens), dtype=torch.float32)
                            .scatter_(-1, data_chunk[..., None], 1)
                            .transpose(0, 1)
                    )
                    gen_samples = torch.transpose(fake_chunk, 0, 1)

                    d_out_real = self.discriminator(real_samples)
                    d_out_fake = self.discriminator(gen_samples)

                    if 'ppo' in self.cfg.DISCRIMINATOR.CNN.loss_type and 'gen' in train_loss:
                        surr1 = ratio * d_out_fake
                        surr2 = ratio_clipped * d_out_fake
                        target = torch.where(d_out_fake > 0, torch.min(surr1, surr2), torch.max(surr1, surr2))
                        temp_gen_loss, temp_dis_loss = get_losses(d_out_real, target,
                                                                  self.cfg.DISCRIMINATOR.CNN.loss_type)
                    else:
                        temp_gen_loss, temp_dis_loss = get_losses(
                            d_out_real, d_out_fake, self.cfg.DISCRIMINATOR.CNN.loss_type
                        )
                    # Regularize discriminator with gradient penalty
                    if "dis" in train_loss and 'gp' in self.cfg.DISCRIMINATOR.CNN.loss_type:
                        temp_gp_loss = self.calc_gradient_penalty(real_samples, gen_samples)

                    if self.cfg.DISCRIMINATOR.backprop_outside:
                        gen_loss += temp_gen_loss.detach()
                        dis_loss += temp_dis_loss.detach()
                        if "dis" in train_loss and 'gp' in self.cfg.DISCRIMINATOR.CNN.loss_type:
                            gp_loss += temp_gp_loss.detach()
                    else:
                        gen_loss += temp_gen_loss
                        dis_loss += temp_dis_loss
                        if "dis" in train_loss and 'gp' in self.cfg.DISCRIMINATOR.CNN.loss_type:
                            gp_loss += temp_gp_loss

                else:
                    raise NotImplementedError


                if self.cfg.DISCRIMINATOR.backprop_outside:
                    if "gen" in train_loss:
                        ((temp_gen_loss.float().mean()) * self.cfg.DISCRIMINATOR.gen_loss_factor / (
                                self.cfg.DISCRIMINATOR.batch_chunk * self.cfg.DISCRIMINATOR.sample_chunks_mem)).backward()
                    if "dis" in train_loss:
                        ((temp_dis_loss.float().mean()) * self.cfg.DISCRIMINATOR.dis_loss_factor / (
                                self.cfg.DISCRIMINATOR.batch_chunk * self.cfg.DISCRIMINATOR.sample_chunks_mem)).backward()

                        if self.cfg.DISCRIMINATOR.type == "bert":
                            if 'gp' in self.cfg.DISCRIMINATOR.BERT.loss_type:
                                ((temp_gp_loss.float().mean()) * self.cfg.DISCRIMINATOR.dis_loss_factor / (
                                        self.cfg.DISCRIMINATOR.batch_chunk * self.cfg.DISCRIMINATOR.sample_chunks_mem)).backward()
                        elif self.cfg.DISCRIMINATOR.type == "cnn":
                            if 'gp' in self.cfg.DISCRIMINATOR.CNN.loss_type:
                                ((temp_gp_loss.float().mean()) * self.cfg.DISCRIMINATOR.dis_loss_factor / (
                                        self.cfg.DISCRIMINATOR.batch_chunk * self.cfg.DISCRIMINATOR.sample_chunks_mem)).backward()  # TODO CNN WGAN-GP
                        else:
                            raise NotImplementedError

                # Reset params for next chunk
                sample_mems = sample_mems.detach()
                seq = [seq[-1]]

            # Reset model parameters
            self.generator.detach_mems_grad = True
            self.generator.reset_length(cache_tgt_len, cache_mem_len)

            #Setup values to return
            if "dis" in train_loss:
                dis_loss = self.cfg.DISCRIMINATOR.dis_loss_factor * dis_loss / self.cfg.DISCRIMINATOR.sample_chunks_mem
                return_dict["dis_loss"] = dis_loss
                if self.cfg.DISCRIMINATOR.type == "bert":
                    if 'gp' in self.cfg.DISCRIMINATOR.BERT.loss_type:
                        return_dict[
                            "gp_loss"] = self.cfg.DISCRIMINATOR.dis_loss_factor * gp_loss / self.cfg.DISCRIMINATOR.sample_chunks_mem
                elif self.cfg.DISCRIMINATOR.type == "cnn":
                    if 'gp' in self.cfg.DISCRIMINATOR.CNN.loss_type:
                        return_dict[
                            "gp_loss"] = self.cfg.DISCRIMINATOR.dis_loss_factor * gp_loss / self.cfg.DISCRIMINATOR.sample_chunks_mem
                else:
                    raise NotImplementedError

            elif "gen" in train_loss:
                gen_loss = self.cfg.DISCRIMINATOR.gen_loss_factor * gen_loss / self.cfg.DISCRIMINATOR.sample_chunks_mem
                return_dict["gen_loss"] = gen_loss

        return return_dict

    def create_bert_model(self, model_name_or_path, loss_type, model_type=None, random_weights=False):

        config_class = BertConfig
        config = config_class.from_pretrained(model_name_or_path, cache_dir=None)

        if model_type == "bert_lm":
            if random_weights:
                print("Starting from random")
                model = BertForSequenceClassification(config=config)
            else:
                model_class = BertForMaskedLM
                model_lm = model_class.from_pretrained(
                    model_name_or_path,
                    from_tf=bool(".ckpt" in model_name_or_path),
                    config=config,
                    cache_dir=None,
                )
                model = BertForSequenceClassification(config=config)
                model.bert = model_lm.bert

        else:
            if random_weights:
                raise NotImplementedError
            model_class = BertForSequenceClassification
            model = model_class.from_pretrained(
                model_name_or_path,
                from_tf=bool(".ckpt" in model_name_or_path),
                config=config,
                cache_dir=None,
            )

        return model.bert if loss_type == "mmd" else model

    def calculate_unfreeze_idx(self, cfg):
        cn, unfreeze_idx, layers = 0, [], []
        for name, param in self.discriminator.named_parameters():
            if name.startswith("bert.embeddings") and not cfg.DISCRIMINATOR.BERT.random_weights:
                pass
            elif name.startswith("bert.encoder.layer") and name.split('.')[3] in cfg.DISCRIMINATOR.BERT.freeze_layers:
                pass
            else:
                unfreeze_idx.append(cn)
            cn += 1

            if name.startswith("bert.encoder.layer"):
                layers.append(name.split('.')[3])

        # check the total number of layers in the BERT >= the number of layers to be freeze
        assert len(layers) >= len(cfg.DISCRIMINATOR.BERT.freeze_layers)

        return unfreeze_idx
