from yacs.config import CfgNode as CN

def get_default_cfg_inference():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    cfg = CN()

    # Event Represnetation Details
    cfg.EVENT = CN()
    cfg.EVENT.event_representation = 'magenta'
    cfg.EVENT.vocab_file_path = '../data/performance_vocab.txt'

    # Model related parameters
    cfg.MODEL = CN()
    cfg.MODEL.model_directory = ''
    cfg.MODEL.memory_length = 100
    cfg.MODEL.src_mem_len = 100
    cfg.MODEL.checkpoint_name = 'checkpoint.pt'
    cfg.MODEL.device = "gpu"
    cfg.MODEL.debug = False

    # Sampling related parameters
    cfg.SAMPLING = CN()
    cfg.SAMPLING.technique = 'topk'
    cfg.SAMPLING.threshold = 32.0
    cfg.SAMPLING.temperature = 0.95

    # Model related parameters
    cfg.GENERATION = CN()
    cfg.GENERATION.generation_length = 100
    cfg.GENERATION.duration_based = False
    cfg.GENERATION.generation_duration = 30  # This duration is based on the time_shift_token in the vocab, which is not
    # exactly time in MIDI because of tempo
    cfg.GENERATION.max_generation_length = 10000  # When flag duration is on, this is maximum generation length

    # Input related parameters
    cfg.INPUT = CN()
    cfg.INPUT.time_extension = True
    cfg.INPUT.conditional_input_melody = ''
    cfg.INPUT.num_conditional_tokens = 100
    cfg.INPUT.conditional_duration = 10

    cfg.INPUT.harmonization = ''
    cfg.INPUT.exclude_bos_token = True
    cfg.INPUT.num_midi_files = 5
    cfg.INPUT.num_empty_tokens_to_ignore = 0

    # Event Representation Details
    cfg.OUTPUT = CN()
    cfg.OUTPUT.output_txt_directory = ''

    cfg.freeze()
    return cfg
