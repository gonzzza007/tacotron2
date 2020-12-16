from tacotron2.text import symbols
global symbols
import json
import numpy as np


class Config:
    # ** Audio params **
    sampling_rate = 22050                        # Sampling rate
    filter_length = 1024                         # Filter length
    hop_length = 256                             # Hop (stride) length
    win_length = 1024                            # Window length
    mel_fmin = 0.0                               # Minimum mel frequency
    mel_fmax = 8000.0                            # Maximum mel frequency
    n_mel_channels = 80                          # Number of bins in mel-spectrograms
    max_wav_value = 32768.0                      # Maximum audiowave value

    # Audio postprocessing params
    snst = 0.00005                               # filter sensitivity
    wdth = 1000                                  # width of filter

    # ** Tacotron Params **
    # Symbols
    n_symbols = len(symbols)                     # Number of symbols in dictionary
    symbols_embedding_dim = 512                  # Text input embedding dimension

    # Speakers
    n_speakers = 128                             # Number of speakers
    speakers_embedding_dim = 16                  # Speaker embedding dimension
    try:
        speaker_coefficients = json.load(open('train/speaker_coefficients.json'))  # Dict with speaker coefficients
    except IOError:
        print("Speaker coefficients dict is not available")
        speaker_coefficients = None

    # Emotions
    use_emotions = True                          # Use emotions
    n_emotions = 15                              # N emotions
    emotions_embedding_dim = 8                   # Emotion embedding dimension
    try:
        emotion_coefficients = json.load(open('train/emotion_coefficients.json'))  # Dict with emotion coefficients
    except IOError:
        print("Emotion coefficients dict is not available")
        emotion_coefficients = None

    # Encoder
    encoder_kernel_size = 5                      # Encoder kernel size
    encoder_n_convolutions = 3                   # Number of encoder convolutions
    encoder_embedding_dim = 512                  # Encoder embedding dimension

    # Attention
    attention_rnn_dim = 1024                     # Number of units in attention LSTM
    attention_dim = 128                          # Dimension of attention hidden representation

    # Attention location
    attention_location_n_filters = 32            # Number of filters for location-sensitive attention
    attention_location_kernel_size = 31          # Kernel size for location-sensitive attention

    # Decoder
    n_frames_per_step = 2                        # Number of frames processed per step
    max_frames = 2000                            # Maximum number of frames for decoder
    decoder_rnn_dim = 1024                       # Number of units in decoder LSTM
    prenet_dim = 256                             # Number of ReLU units in prenet layers
    max_decoder_steps = int(max_frames / n_frames_per_step)  # Maximum number of output mel spectrograms
    gate_threshold = 0.5                         # Probability threshold for stop token
    p_attention_dropout = 0.1                    # Dropout probability for attention LSTM
    p_decoder_dropout = 0.1                      # Dropout probability for decoder LSTM
    decoder_no_early_stopping = False            # Stop decoding once all samples are finished

    # Postnet
    postnet_embedding_dim = 512                  # Postnet embedding dimension
    postnet_kernel_size = 5                      # Postnet kernel size
    postnet_n_convolutions = 5                   # Number of postnet convolutions

    # Optimization
    mask_padding = False                         # Use mask padding
    use_loss_coefficients = True                 # Use balancing coefficients
    # Loss scale for coefficients
    if emotion_coefficients is not None and speaker_coefficients is not None:
        loss_scale = 1.5 / (np.mean(list(speaker_coefficients.values())) * np.mean(list(emotion_coefficients.values())))
    else:
        loss_scale = None

    # ** Waveglow params **
    n_flows = 12                                 # Number of steps of flow
    n_group = 8                                  # Number of samples in a group processed by the steps of flow
    n_early_every = 4                            # Determines how often (i.e., after how many coupling layers) a number of channels (defined by --early-size parameter) are output to the loss function
    n_early_size = 2                             # Number of channels output to the loss function
    wg_sigma = 1.0                               # Standard deviation used for sampling from Gaussian
    segment_length = 4000                        # Segment length (audio samples) processed per iteration
    wn_config = dict(
        n_layers=8,                              # Number of layers in WN
        kernel_size=3,                           # Kernel size for dialted convolution in the affine coupling layer (WN)
        n_channels=512                           # Number of channels in WN
    )

    # ** Script args **
    model_name = "Tacotron2"
    output_directory = "logs"                    # Directory to save checkpoints
    log_file = "nvlog.json"                      # Filename for logging

    anneal_steps = [500, 1000, 1500]             # Epochs after which decrease learning rate
    anneal_factor = 0.1                          # Factor for annealing learning rate

    tacotron2_checkpoint = 'pretrained/t2_fp32_torch'   # Path to pre-trained Tacotron2 checkpoint for sample generation
    waveglow_checkpoint = 'pretrained/wg_fp32_torch'    # Path to pre-trained WaveGlow checkpoint for sample generation
    restore_from = ''                                        # Checkpoint path to restore from

    # Training params
    epochs = 1501                                # Number of total epochs to run
    epochs_per_checkpoint = 50                   # Number of epochs per checkpoint
    seed = 1234                                  # Seed for PyTorch random number generators
    dynamic_loss_scaling = True                  # Enable dynamic loss scaling
    amp_run = False                              # Enable AMP (FP16) # TODO: Make it work
    cudnn_enabled = True                         # Enable cudnn
    cudnn_benchmark = False                      # Run cudnn benchmark

    # Optimization params
    use_saved_learning_rate = False
    learning_rate = 1e-3                         # Learning rate
    weight_decay = 1e-6                          # Weight decay
    grad_clip_thresh = 1                         # Clip threshold for gradients
    batch_size = 32                              # Batch size per GPU
    # batch_size = 64                              # Batch size per GPU

    # Dataset
    load_mel_from_dist = False                   # Loads mel spectrograms from disk instead of computing them on the fly
    text_cleaners = ['english_cleaners']         # Type of text cleaners for input text
    training_files = 'train/train.txt'          # Path to training filelist
    validation_files = 'train/val.txt'          # Path to validation filelist

    dist_url = 'tcp://localhost:23456'           # Url used to set up distributed training
    group_name = "group_name"                    # Distributed group name
    dist_backend = "nccl"                        # Distributed run backend

    # Sample phrases
    phrases = {
        'speaker_ids': [0, 2],
        'texts': [
            'Hello, how are you doing today?',
            'I would like to eat a Hamburger.',
            'Hi.',
            'I would like to eat a Hamburger. Would you like to join me?',
            'Do you have any hobbies?'
        ]
    }


class PreprocessingConfig:
    cpus = 8                                    # Amount of cpus for parallelization
    sr = 22050                                   # sampling ratio for audio processing
    top_db = 60                                  # level to trim audio
    limit_by = 'Actor_000'                       # speaker to measure text_limit, dur_limit
    minimum_viable_dur = 0.05                    # min duration of audio
    text_limit = None                            # max text length (used by default)
    dur_limit = None                             # max audio duration (used by default)
    n = 15000                                    # max size of training dataset per speaker
    start_from_preprocessed = True              # load data.csv - should be in output_directory

    output_directory = 'train'
    

    ###########################################################
    # emotion_present - True by default
    # process_audio - True by default
    # speaker_id - autoincrement from - 0 
    ###########################################################
    data = [
        { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_000', 'process_audio': True },
        { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_001', 'process_audio': True },
        { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_002', 'process_audio': True }


        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_004' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_005' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_006' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_007' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_008' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_009' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_010' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_011' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_012' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_013' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_014' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_015' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_016' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_017' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_018' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_019' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_020' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_021' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_022' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_023' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_024' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_025' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_026' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_027' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_028' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_029' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_030' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_031' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_032' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_033' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_034' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_035' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_036' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_037' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_038' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_039' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_040' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_041' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_042' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_043' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_044' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_045' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_046' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_047' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_048' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_049' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_050' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_051' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_052' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_053' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_054' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_055' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_056' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_057' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_058' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_059' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_060' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_061' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_062' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_063' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_064' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_065' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_066' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_067' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_068' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_069' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_070' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_071' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_072' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_073' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_074' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_075' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_076' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_077' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_078' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_079' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_080' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_081' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_082' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_083' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_084' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_085' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_086' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_087' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_088' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_089' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_090' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_091' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_092' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_093' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_094' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_095' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_096' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_097' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_098' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_099' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_100' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_101' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_102' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_103' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_104' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_105' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_106' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_107' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_108' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_109' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_110' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_111' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_112' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_113' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_114' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_115' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_116' },
        # { 'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_117' }


        # {
        #     'path': 'C:\\Users\\gonza\\SOD\\DATASET\\Actor_04',
        #     'speaker_id': 3,
        #     'metadata_file': 'metadata.csv',
        #     'process_audio': True,
        #     'emotion_present': True
        # },

        
    ]

    emo_id_map = {
        'neutral-normal': 0,
        'calm-normal': 1,
        'calm-strong': 2,
        'happy-normal': 3,
        'happy-strong': 4,
        'sad-normal': 5,
        'sad-strong': 6,
        'angry-normal': 7,
        'angry-strong': 8,
        'fearful-normal': 9,
        'fearful-strong': 10,
        'disgust-normal': 11,
        'disgust-strong': 12,
        'surprised-normal': 13,
        'surprised-strong': 14
    }