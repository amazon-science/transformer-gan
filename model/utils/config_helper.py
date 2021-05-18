from yacs.config import CfgNode as CN

def model(cfg):
    # For model
    cfg.MODEL = CN()
    cfg.MODEL.num_layers = 6
    cfg.MODEL.num_heads = 10
    cfg.MODEL.units = 500
    cfg.MODEL.inner_size = 1000
    cfg.MODEL.dropout = 0.1
    cfg.MODEL.tie_embedding = True
    cfg.MODEL.tie_proj = False
    cfg.MODEL.attention_dropout = 0.1
    cfg.MODEL.pre_lnorm = False
    cfg.MODEL.clamp_len = -1
    cfg.MODEL.same_length = False
    return cfg

def train(cfg):
    # For training
    cfg.TRAIN = CN()
    cfg.TRAIN.load_from_previous = "Null"
    cfg.TRAIN.batch_size = 200
    cfg.TRAIN.batch_chunk = 1
    cfg.TRAIN.tgt_length = 500
    cfg.TRAIN.mem_length = 50
    cfg.TRAIN.seed = 1111
    cfg.TRAIN.optim = "adam"
    cfg.TRAIN.lr = 0.00025 / 4.0
    cfg.TRAIN.lr_min = 0.0
    cfg.TRAIN.scheduler = "cosine"
    cfg.TRAIN.warmup_step = 0
    cfg.TRAIN.decay_rate = 0.5
    cfg.TRAIN.patience = 10
    cfg.TRAIN.clip = 0.25
    cfg.TRAIN.max_step = 200000
    cfg.TRAIN.log_interval = 200
    cfg.TRAIN.eval_interval = 4000
    cfg.TRAIN.pad_type = "model"  # model or anything else
    cfg.TRAIN.use_mle = True
    cfg.TRAIN.random_crop = False
    cfg.TRAIN.replace_start_with_pad = False
    cfg.TRAIN.weight_decay = 0.0  # Weight decay for adam or lamb
    cfg.TRAIN.append_note_status = False  # Append status to event representation
    return cfg


def discriminator(cfg):
    # For discriminator
    # Discriminator related (used only if)
    cfg.DISCRIMINATOR = CN()
    cfg.DISCRIMINATOR.start_iter = 100  # To control when we start training critic
    cfg.DISCRIMINATOR.dis_loss_freq = 50  # How often to use loss from discriminator
    cfg.DISCRIMINATOR.gen_loss_freq = 10

    cfg.DISCRIMINATOR.eval_loss_freq = 10  # How often to use loss from discriminator during eval
    cfg.DISCRIMINATOR.freeze_discriminator = True
    cfg.DISCRIMINATOR.truncate_backprop = False  # while sampling do not propagate gradients beyond current token
    cfg.DISCRIMINATOR.sample_chunks_mem = 1
    cfg.DISCRIMINATOR.beta_max = 100.  # TODO: temperature decay
    cfg.DISCRIMINATOR.adapt = 'no'
    cfg.DISCRIMINATOR.type = "Null"  # or cnn or Null for no discriminator or 'bert' for BERT discriminator
    cfg.DISCRIMINATOR.dis_steps = 1  # dis_step per gen_step (default 1 for bert and 5 for cnn)
    cfg.DISCRIMINATOR.tgt_len = 64
    cfg.DISCRIMINATOR.mem_len = 64
    cfg.DISCRIMINATOR.gen_loss_factor = 30  # Multiplying factor for mmd/gan loss component in generator
    cfg.DISCRIMINATOR.dis_loss_factor = 1  # Multiplying factor for mmd/gan loss component in discriminator
    cfg.DISCRIMINATOR.batch_chunk = 1
    cfg.DISCRIMINATOR.context_len = 5  # Randomly sample context length tokens from real data and use as context.
    cfg.DISCRIMINATOR.backprop_outside = True
    cfg.DISCRIMINATOR.src_mem_len = 200

    # If 0 uses first token in real data
    cfg.DISCRIMINATOR.gen_scheduler = "constant"
    cfg.DISCRIMINATOR.gen_lr_min = 0.0
    cfg.DISCRIMINATOR.gen_warmup_step = 0
    cfg.DISCRIMINATOR.gen_decay_rate = 0.5
    cfg.DISCRIMINATOR.gen_patience = 10
    cfg.DISCRIMINATOR.gen_lr = 0.00025 / 4.0

    cfg.DISCRIMINATOR.dis_scheduler = "constant"
    cfg.DISCRIMINATOR.dis_lr_min = 0.0
    cfg.DISCRIMINATOR.dis_warmup_step = 0
    cfg.DISCRIMINATOR.dis_decay_rate = 0.5
    cfg.DISCRIMINATOR.dis_patience = 10
    cfg.DISCRIMINATOR.dis_lr = 0.00025 / 4.0

    # Bert params
    cfg.DISCRIMINATOR.BERT = CN()
    cfg.DISCRIMINATOR.BERT.learning_rate = 1e-5  # Decrease learning rate since we're fine tuning
    cfg.DISCRIMINATOR.BERT.weight_decay = 0.0
    cfg.DISCRIMINATOR.BERT.adam_epsilon = 1e-8
    cfg.DISCRIMINATOR.BERT.max_grad_norm = 1.0
    cfg.DISCRIMINATOR.BERT.model_type = "bert_lm"  # or "bert_cls"
    cfg.DISCRIMINATOR.BERT.loss_type = "rsgan"  # or 'standard’,'JS', 'KL', 'hinge', 'tv', 'rsgan', 'wgan-gp', "mmd", 'ppo', 'ppo-gp'
    cfg.DISCRIMINATOR.BERT.model_path = "../BERT/checkpoint-1969000"
    cfg.DISCRIMINATOR.BERT.freeze_layers = []  # Total layers ['0', '1', '2', '3', '4']
    cfg.DISCRIMINATOR.BERT.random_weights = False  # only implemented for bert_lm

    # CNN params (Relgan)
    cfg.DISCRIMINATOR.CNN = CN()
    cfg.DISCRIMINATOR.CNN.learning_rate = 1e-4
    cfg.DISCRIMINATOR.CNN.embed_dim = 64
    cfg.DISCRIMINATOR.CNN.hidden_dim = 64
    cfg.DISCRIMINATOR.CNN.num_rep = 64
    cfg.DISCRIMINATOR.CNN.init = "uniform"
    cfg.DISCRIMINATOR.CNN.loss_type = "rsgan"  # or 'standard’,'JS', 'KL', 'hinge', 'tv', 'rsgan', 'wgan-gp', "mmd", "ppo-gp"
    return cfg

def metric(cfg):
    # Metrics
    cfg.METRICS = CN()
    cfg.METRICS.use_bleu = False  # outdated
    cfg.METRICS.use_self_bleu = False  # outdated
    cfg.METRICS.CLASSIFIER = CN()
    cfg.METRICS.CLASSIFIER.use_classifier = False
    cfg.METRICS.CLASSIFIER.gen_batch_size = 128
    cfg.METRICS.CLASSIFIER.gen_seq_len = 2048
    cfg.METRICS.CLASSIFIER.gen_num_samples = 256
    cfg.METRICS.CLASSIFIER.block_size = 128  # For training classifier
    cfg.METRICS.CLASSIFIER.bert_batch_size = 20  # For passing into bert
    cfg.METRICS.CLASSIFIER.model_path = "../BERT/checkpoint-1969000"
    return cfg

def init(cfg):
    # For initialization
    cfg.INITIALIZER = CN()
    cfg.INITIALIZER.base_init = ["normal", 0.01]
    cfg.INITIALIZER.embed_init = ["normal", 0.01]

    # For evaluation
    cfg.EVALUATE = CN()
    cfg.EVALUATE.batch_size = 10
    cfg.EVALUATE.tgt_length = 128
    cfg.EVALUATE.mem_length = 128
    # Event type related
    cfg.DATASET = CN()
    cfg.DATASET.event_type = "magenta"  # or 'newevent'
    cfg.DATASET.trim_padding = False

    # Classifier related
    cfg.PPO = CN()  # For ppo loss type
    cfg.PPO.dis_D_lr = 0.00025 / 4.0
    cfg.PPO.dis_D_update_D0_freq = 20  # Should be multiple of gen_loss_freq
    cfg.PPO.dis_D_type = "bert"  # bert or cnn
    cfg.PPO.clip_param = 0.4
    cfg.PPO.dis_D_num_rep = 1

    # For Problem Type
    cfg.PROBLEM = CN()
    cfg.PROBLEM.type = 'Null' # time extension: Null
    cfg.PROBLEM.melody_len = 1024
    return cfg

def get_default_cfg_training():
    cfg = CN()
    cfg = init(cfg)
    cfg = model(cfg)
    cfg = train(cfg)
    cfg = discriminator(cfg)
    cfg = metric(cfg)
    cfg.freeze()
    return cfg
