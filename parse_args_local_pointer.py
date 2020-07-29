
import os,sys
import argparse
import torch
from argparse import Namespace
import numpy as np
import random

def interpret_args():
    """ Interprets the command line arguments, and returns a dictionary. """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--seed', default=1234, type=int, help='random seed')
    arg_parser.add_argument('--cuda', action='store_true', help='use gpu')
    arg_parser.add_argument('--use_bert', action='store_true', help='use bert')
    arg_parser.add_argument('--fine_tune_bert', action='store_true', help='finetune bert')

    arg_parser.add_argument('--discourse_level_lstm', action='store_true', help='discourse state,store question context')

    arg_parser.add_argument('--print_info', action='store_true', help='print')

    arg_parser.add_argument('--use_schema_encoder_2', action='store_true', help='schema-utterance attention BiLSTM')
    arg_parser.add_argument('--use_encoder_attention',action='store_true', help='schema-utterance attention')
    arg_parser.add_argument('--stage_epoch',default=[5,16,27,36],help='stage epoch')

    arg_parser.add_argument('--use_copy_switch', type=bool, default=False)
    arg_parser.add_argument('--interaction_level', action='store_true', help='use interaction_level to train , that is batch_size = 1')

    arg_parser.add_argument('--lr_decay', default=5, type=float,
                            help='decay rate of learning rate')

    arg_parser.add_argument('--column_pointer', action='store_true', help='use column pointer')
    arg_parser.add_argument('--loss_epoch_threshold', default=20, type=int, help='loss epoch threshold')
    arg_parser.add_argument('--sketch_loss_coefficient', default=0.2, type=float, help='sketch loss coefficient')
    arg_parser.add_argument('--sentence_features', action='store_true', help='use sentence features')
    arg_parser.add_argument('--model_name', choices=['transformer', 'rnn', 'table', 'sketch'], default='rnn',
                            help='model name')

    arg_parser.add_argument('--use_query_attention', type=bool, default=False)
    arg_parser.add_argument('--lstm', choices=['lstm', 'lstm_with_dropout', 'parent_feed'], default='lstm')

    arg_parser.add_argument('--load_model', default=None, type=str, help='load a pre-trained model')
    arg_parser.add_argument('--glove_embed_path', default="glove.42B.300d.txt", type=str)

    arg_parser.add_argument('--batch_size', default=64, type=int, help='batch size')

    arg_parser.add_argument('--encoder_state_size', type=int, default=300,help='discourse state size')
    arg_parser.add_argument('--encoder_num_layers', type=int, default=1, help='encoder num layers')
    arg_parser.add_argument('--use_schema_encoder', action='store_true', help='use schema encoder')

    arg_parser.add_argument('--beam_size', default=5, type=int, help='beam size for beam search')
    arg_parser.add_argument('--embed_size', default=300, type=int, help='size of word embeddings')
    arg_parser.add_argument('--col_embed_size', default=300, type=int, help='size of word embeddings')
    arg_parser.add_argument('--use_schema_self_attention', type=bool, default=False)


    arg_parser.add_argument('--positional_embedding_size', type=int, default=50)
    arg_parser.add_argument('--input_embedding_size', type=int, default=300)

    arg_parser.add_argument('--maximum_queries', type=int, default=1)
    arg_parser.add_argument('--bert_type_abb', default='uS', type=str, help='bert model type')
    arg_parser.add_argument('--action_embed_size', default=128, type=int, help='size of word embeddings')
    arg_parser.add_argument('--type_embed_size', default=128, type=int, help='size of word embeddings')
    arg_parser.add_argument('--hidden_size', default=100, type=int, help='size of LSTM hidden states')
    arg_parser.add_argument('--att_vec_size', default=100, type=int, help='size of attentional vector')
    arg_parser.add_argument('--dropout', default=0.3, type=float, help='dropout rate')
    arg_parser.add_argument('--word_dropout', default=0.2, type=float, help='word dropout rate')

    arg_parser.add_argument('--use_previous_query', type=bool, default=False)
    arg_parser.add_argument('--state_positional_embeddings', type=bool, default=False)
    arg_parser.add_argument('--use_utterance_attention', type=bool, default=False)
    arg_parser.add_argument('--dropout_amount', type=float, default=0.5)
    # readout layer
    arg_parser.add_argument('--no_query_vec_to_action_map', default=False, action='store_true')
    arg_parser.add_argument('--readout', default='identity', choices=['identity', 'non_linear'])
    arg_parser.add_argument('--query_vec_to_action_diff_map', default=False, action='store_true')

    arg_parser.add_argument('--column_att', choices=['dot_prod', 'affine'], default='affine')

    arg_parser.add_argument('--decode_max_time_step', default=40, type=int, help='maximum number of time steps used '
                                                                         'in decoding and sampling')

    arg_parser.add_argument('--save_to', default='model', type=str, help='save trained model to')
    arg_parser.add_argument('--use_small', action='store_true',
                            help='If set, use small data; used for fast debugging.')
    arg_parser.add_argument('--clip_grad', default=5., type=float, help='clip gradients')
    arg_parser.add_argument('--max_epoch', default=-1, type=int, help='maximum number of training epoches')
    arg_parser.add_argument('--optimizer', default='Adam', type=str, help='optimizer')
    arg_parser.add_argument('--initial_lr', default=0.001, type=float, help='learning rate')
    arg_parser.add_argument('--lr_bert', default=1e-5, type=float, help='bert learning rate')
    arg_parser.add_argument('--decode_max_time_step', default=40, type=int, help='maximum number of time steps used '
                                                                                      'in decoding and sampling')

    arg_parser.add_argument('--dataset', default="./data", type=str)
    arg_parser.add_argument('--maximum_utterances', type=int, default=5)
    arg_parser.add_argument('--epoch', default=50, type=int, help='Maximum Epoch')
    arg_parser.add_argument('--save', default='./', type=str,
                            help="Path to save the checkpoint and logs of epoch")

    return arg_parser.parse_args()


def get_local_args():

    args = Namespace(
        dataset='process_data2',
        epoch=21,
        loss_epoch_threshold=10,
        sketch_loss_coefficient=0.5,
        beam_size=1,

        seed=90,
        save='model_local_pointer',

        predict_save = 'predict_sparc',

        logfile = 'logfile.log',
        log_pred_gt='log_pred_gt.log',
        embed_size=300,
        sentence_features=True,
        column_pointer=True,
        hidden_size=300,

        att_vec_size=300,

        train_evaluation_size = 400,

        use_copy_switch = True,
        decode_max_time_step = 40,
        model_name='rnn',

        #0.001 -> 0.0002(2e-4) -> 0.00004(4e-5) -> 0.000008(8e-6)
        #0.001 -> 0.0005 -> 0.00025 -> 0.000125(1.25e-4) -> 0.0000625(6.25e-5) -> 0.00003125(3.125e-5) -> 0.000015625 
        #stage_epoch=[10,18,27,36,42,49,50,55],
        stage_epoch = [5,10,15,18],
        maximum_queries = 1,
        positional_embedding_size = 50,
        input_embedding_size = 300,
        dropout_amount = 0.3,
        lstm='lstm',
        batch_size=64,
        col_embed_size=300,
        action_embed_size=128,
        type_embed_size=128,

        value_max_len = 10,
        dropout=0.1,
        word_dropout=0.2,
        maximum_utterances = 6,
        
        use_column_pointer = True,
        use_query_attention = True,
        use_previous_query = True,
        state_positional_embeddings = True,
        use_utterance_attention = True,
        # readout layer
        no_query_vec_to_action_map=False,
        readout='identity',
        query_vec_to_action_diff_map=False,
        use_schema_self_attention = True,
        column_att='affine',

        save_to='model',

        clip_grad=5.,
        max_epoch=-1,
        optimizer='Adam',
        initial_lr=0.001,
        lr_decay=10,
        lr_bert=1e-5,

        best_model_name = 'best_w.pth',
        current_model_name='current_w.pth',

        bert_type_abb='uS',
        fine_tune_bert = True,
        use_bert = True,

        interaction_level = True,
        encoder_num_layers = 1,
        encoder_state_size=300,

        use_schema_encoder_2 = True,
        use_encoder_attention = True,
        use_schema_encoder = True,
        load_model=False,
        use_small=False,
        print_info = True,
        discourse_level_lstm = True,

        resume =False,
        cuda=True,
        toy = False

    )

    return args

def init_config(arg_param):

    torch.manual_seed(arg_param.seed)
    if arg_param.cuda:
        torch.cuda.manual_seed(arg_param.seed)
    np.random.seed(arg_param.seed)
    random.seed(arg_param.seed)

    return arg_param
