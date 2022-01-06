import tensorflow as tf
from text import symbols
from pprint import pprint

def create_hparams(hparams_string=None, hparams_json=None, verbose=True):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        training_stage='train_style_extractor',#['train_text_encoder','train_style_extractor','train_style_attention','train_refine_layernorm']
        full_refine=False,
        ################################
        # Experiment Parameters        #
        ################################
        epochs=500,
        iters=1000000,
        iters_per_checkpoint=5000,
        log_per_checkpoint=1,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        numberworkers=8,
        ignore_layers=['embedding.weight'],

        ################################
        # Data Parameters             #
        ################################
        load_mel=True,
        training_files='../../../spk_ttsdatafull_libri500_unpacked/training_with_mel_frame.txt',
        mel_dir='../../../spk_ttsdatafull_libri500_unpacked/',
        text_cleaners=['english_cleaners'],
        is_partial_refine=False,
        is_refine_style=False,
        use_GAN=False,
        GAN_type='wgan-gp',#['lsgan', 'wgan-gp']
        GAN_alpha=1.0,
        GP_beata=10.0,
        Generator_pretrain_step=1,
        add_noise=False,

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        num_mels=80,
        num_freq=1025,
        min_mel_freq=0,
        max_mel_freq=8000,
        sample_rate=16000,
        frame_length_ms=50,
        frame_shift_ms=12.5,
        preemphasize=0.97,
        min_level_db=-100,
        ref_level_db=0,  # suggest use 20 for griffin-lim and 0 for wavenet
        max_abs_value=4,
        symmetric_specs=True,  # if true, suggest use 4 as max_abs_value

        # Eval:
        griffin_lim_iters=60,
        power=1.5,  # Power to raise magnitudes to prior to Griffin-Lim
        threshold=0.5,  # for stop token
        minlenratio=0.0,  # Minimum length ratio in inference.
        maxlenratio=50.0,  # Maximum length ratio in inference.

        use_phone=True,
        phone_set_file="../../../spk_ttsdatafull_libri500_unpacked/phone_set.json",
        n_symbols=5000,  # len(symbols),
        embed_dim=512,  # Dimension of character embedding.

        pretrained_model=None,

        # VQVAE
        use_vqvae=False,
        aux_encoder_kernel_size=3,
        aux_encoder_n_convolutions=2,
        aux_encoder_embedding_dim=512,
        speaker_embedding_dim=256,
        commit_loss_weight=1.0, # Contribution of commitment loss, between 0.1 and 2.0 (default: 1.0)

        eprenet_conv_layers=3,  # Number of encoder prenet convolution layers.
        eprenet_conv_chans=512,  # Number of encoder prenet convolution channels.
        eprenet_conv_filts=5,  # Filter size of encoder prenet convolution.
        dprenet_layers=2,  # Number of decoder prenet layers.
        dprenet_units=256,  # Number of decoder prenet hidden units.

        positionwise_layer_type="linear",  # FFN or conv or (conv+ffn) in encoder after self-attention
        positionwise_conv_kernel_size=1,  # Filter size of conv

        elayers=6,  # Number of encoder layers.
        eunits=1536,  # Number of encoder hidden units.
        adim=384,  # Number of attention transformation dimensions.
        aheads=4,  # Number of heads for multi head attention.
        dlayers=6,  # Number of decoder layers.
        dunits=1536,  # Number of decoder hidden units.
        duration_predictor_layers=2,
        duration_predictor_chans=384,
        duration_predictor_kernel_size=3,
        use_gaussian_upsampling=False,

        postnet_layers=5,  # Number of postnet layers.
        postnet_chans=512,  # Number of postnet channels.
        postnet_filts=5,  # Filter size of postnet.

        use_scaled_pos_enc=True,  # Whether to use trainable scaled positional encoding.
        use_batch_norm=True,  # Whether to use batch normalization in posnet.
        encoder_normalize_before=True,  # Whether to perform layer normalization before encoder block.
        decoder_normalize_before=True,  # Whether to perform layer normalization before decoder block.
        encoder_concat_after=False,  # Whether to concatenate attention layer's input and output in encoder.
        decoder_concat_after=False,  # Whether to concatenate attention layer's input and output in decoder.
        reduction_factor=1,  # Reduction factor.

        is_multi_speakers=True,
        is_spk_layer_norm=True,
        pretrained_spkemb_dim=512,
        n_speakers=8000,
        spk_embed_dim=128,  # Number of speaker embedding dimenstions.
        spk_embed_integration_type="concat",  # concat or add, How to integrate speaker embedding.

        use_ssim_loss=True,
        use_f0=False,
        log_f0=False,
        f0_joint_train=False,
        f0_alpha=0.1,
        stop_gradient_from_pitch_predictor=False,
        pitch_predictor_layers=2,
        pitch_predictor_chans=384,
        pitch_predictor_kernel_size=3,
        pitch_predictor_dropout=0.5,
        pitch_embed_kernel_size=9,
        pitch_embed_dropout=0.5,

        is_multi_styles=False,
        n_styles=6,
        style_embed_dim=128,  # Number of style embedding dimenstions.
        style_embed_integration_type="concat",  # concat or add, How to integrate style embedding.
        style_vector_type='mha',#gru or mha, How to generate style vector.
        style_query_level='sentence',#phone or sentence

        # value: pytorch, xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal
        transformer_init="pytorch",  # How to initialize transformer parameters.
        initial_encoder_alpha=1.0,
        initial_decoder_alpha=1.0,

        transformer_enc_dropout_rate=0.1,  # Dropout rate in encoder except attention & positional encoding.
        transformer_enc_positional_dropout_rate=0.1,  # Dropout rate after encoder positional encoding.
        transformer_enc_attn_dropout_rate=0.1,  # Dropout rate in encoder self-attention module.
        transformer_dec_dropout_rate=0.1,  # Dropout rate in decoder except attention & positional encoding.
        transformer_dec_positional_dropout_rate=0.1,  # Dropout rate after decoder positional encoding.
        transformer_dec_attn_dropout_rate=0.1,  # Dropout rate in deocoder self-attention module.
        transformer_enc_dec_attn_dropout_rate=0.1,  # Dropout rate in encoder-deocoder attention module.
        duration_predictor_dropout_rate=0.1,
        eprenet_dropout_rate=0.5,  # Dropout rate in encoder prenet.
        dprenet_dropout_rate=0.5,  # Dropout rate in decoder prenet.
        postnet_dropout_rate=0.5,  # Dropout rate in postnet.

        use_masking=True,  # Whether to apply masking for padded part in loss calculation.
        use_weighted_masking=False,  # Whether to apply weighted masking in loss calculation.
        bce_pos_weight=1.0,  # Positive sample weight in bce calculation (only for use_masking=true).

        loss_type="L2",   # L1, L2, L1+L2, How to calculate loss.
        # Reference:
        # Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention
        # https://arxiv.org/abs/1710.08969

        use_gst=False,
        use_mutual_information=False,
        mutual_information_lambda=0.1,
        mi_loss_type='unbias',#['bias','unbias']
        style_extractor_presteps=300000,
        choosestl_steps=100000,
        gst_train_att=False,
        att_name='100k_noshuffle_gru',
        shuffle=False,
        gst_reference_encoder='multiheadattention',#'multiheadattention' or 'convs'
        gst_reference_encoder_mha_layers=4,
        gst_tokens=10,
        gst_heads=4,
        gst_conv_layers=6,
        gst_conv_chans_list=(32, 32, 64, 64, 128, 128),
        gst_conv_kernel_size=3,
        gst_conv_stride=2,
        gst_gru_layers=1,
        gst_gru_units=128,

        step_use_predicted_dur=20000,

        ################################
        # Optimization Hyperparameters #
        ################################
        learning_rate_decay_scheme='noam',
        use_saved_learning_rate=True,
        warmup_steps=10000,  # Optimizer warmup steps.
        decay_steps=12500, # halves the learning rate every 12.5k steps
        decay_rate=0.5,  # learning rate decay rate
        # decay_end=300000,
        # decay_rate=0.01,
        initial_learning_rate=0.5,  # Initial value of learning rate.
        final_learning_rate=1e-5,

        weight_decay=1e-6,
        grad_clip_thresh=1.0,

        batch_criterion='utterance',
        batch_size=2,
        mask_padding=True  # set model's padded outputs to padded values
    )

    if hparams_json:
        print('Parsing hparams in json  # {}'.format(hparams_json))
        with open(hparams_json) as json_file:
            hparams.parse_json(json_file.read())

    if hparams_string:
        print('Parsing command line hparams  # {}'.format(hparams_string))
        hparams.parse(hparams_string)

    # if hparams.use_phone:
    #     from text.phones import Phones
    #     phone_class = Phones(hparams.phone_set_file)
    #     hparams.n_symbols = len(phone_class._symbol_to_id)
    #     del phone_class

    if verbose:
        print('Final parsed hparams:')
        pprint(hparams.values())

    return hparams
