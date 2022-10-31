"""
A custom implementation of Show-Attend-&-Tell for ArtEmis: Affective Language for Visual Art

The MIT License (MIT)
Originally created in early 2020, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""

from torch import nn
from .resnet_encoder import ResnetEncoder
from .attentive_decoder import AttentiveDecoder


def describe_model(vocab, args):
    """ Describe the architecture of a SAT speaker with a resnet encoder.
    :param vocab:
    :param args:
    :return:
    """
    word_embedding = nn.Embedding(len(vocab), args.word_embedding_dim, padding_idx=vocab.pad_token_id)

    encoder = ResnetEncoder(args.vis_encoder, adapt_image_size=args.atn_spatial_img_size).unfreeze()
    encoder_out_dim = encoder.embedding_dimension()

    emo_ground_dim = 0
    emo_projection_net = None
    if args.use_emo_grounding:
        emo_in_dim = args.emo_grounding_dims[0]
        emo_ground_dim = args.emo_grounding_dims[1]
        # obviously one could use more complex nets here instead of using a "linear" layer.
        # in my estimate, this is not going to be useful:)
        emo_projection_net = nn.Sequential(*[nn.Linear(emo_in_dim, emo_ground_dim), nn.ReLU()])

    lang_ground_dim = 0
    lang_projection_net = None
    if args.use_lang_grounding:
        lang_in_dim = args.lang_grounding_dims[0]
        lang_ground_dim = args.lang_grounding_dims[1]
        lang_projection_net = nn.Sequential(*[nn.Linear(lang_in_dim, lang_ground_dim), nn.ReLU()])

    decoder = AttentiveDecoder(word_embedding,
                               args.rnn_hidden_dim,
                               encoder_out_dim,
                               args.attention_dim,
                               vocab,
                               dropout_rate=args.dropout_rate,
                               teacher_forcing_ratio=args.teacher_forcing_ratio,
                               auxiliary_net=emo_projection_net,
                               auxiliary_dim=emo_ground_dim,
                               language_net=lang_projection_net,
                               language_dim=lang_ground_dim)

    print(decoder.uses_lang_data, decoder.uses_aux_data)

    model = nn.ModuleDict({'encoder': encoder, 'decoder': decoder})
    return model
