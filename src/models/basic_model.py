"""
# @Time    : 2019/12/09
# @Author  : xiaxia.wang
# @File    : src/models/basic_model.py
# @Software: PyCharm
"""
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from src.rule import semQL as define_rule
from .bert import utils_bert
from . import torch_utils
from .encoder import Encoder



class BasicModel(nn.Module):
    def __init__(self,args):
        super(BasicModel,self).__init__()
        self.args = args
        if args.use_bert:
            self.model_bert, self.tokenizer, self.bert_config = utils_bert.get_bert(args)

        self.device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
        #TODO 会话级别编码
        if args.use_bert:
                encoder_input_size = self.bert_config.hidden_size
        if args.discourse_level_lstm:
            encoder_input_size += args.encoder_state_size // 2
        encoder_output_size = args.encoder_state_size
        self.utterance_encoder = Encoder(args.encoder_num_layers, encoder_input_size, encoder_output_size)

        # Positional embedder for utterances
        attention_key_size = args.encoder_state_size
        # TODO 300
        if args.state_positional_embeddings:
            ##TODo 加上轮次位置，300+50
            attention_key_size += args.positional_embedding_size
            # TODO positional_embedding_size : 50
            init_tensor = torch.empty(args.maximum_utterances, args.positional_embedding_size).uniform_(-0.1, 0.1)
            self.positional_embedder = nn.Embedding.from_pretrained(init_tensor,freeze=False)

        if args.discourse_level_lstm:
            self.discourse_lstms = torch_utils.create_multilayer_lstm_params(1, args.encoder_state_size,
                                                                             args.encoder_state_size // 2, "LSTM-t")
            self.initial_discourse_state = torch_utils.add_params(tuple([args.encoder_state_size // 2]),
                                                                  "V-turn-state-0")
            # torch.Size([150]) True
            # print(self.initial_discourse_state.size(),self.initial_discourse_state.requires_grad)

        self.schema_attention_key_size = args.encoder_state_size

        self.dropout_ratio = 0.


    def padding_sketch(self, sketch):
        padding_result = []
        for action in sketch:
            padding_result.append(action)
            if type(action) == define_rule.N:
                for _ in range(action.id_c + 1):
                    padding_result.append(define_rule.A(0))
                    padding_result.append(define_rule.TC(0))
            elif type(action) == define_rule.Filter and 'A' in action.production:
                padding_result.append(define_rule.A(0))
                padding_result.append(define_rule.TC(0))
            elif type(action) == define_rule.Order or type(action) == define_rule.Sup:
                padding_result.append(define_rule.A(0))
                padding_result.append(define_rule.TC(0))

        return padding_result

    def _add_positional_embeddings(self, hidden_states, utterances, group=False):
        grouped_states = []

        start_index = 0
        for utterance in utterances:
            grouped_states.append(hidden_states[start_index:start_index + len(utterance)])
            start_index += len(utterance)
        assert len(hidden_states) == sum([len(seq) for seq in grouped_states]) == sum([len(utterance) for utterance in utterances])

        new_states = []
        flat_sequence = []
        # print(utterances)
        num_utterances_to_keep = min(self.args.maximum_utterances, len(utterances))
        for i, (states, utterance) in enumerate(zip(
                grouped_states[-num_utterances_to_keep:], utterances[-num_utterances_to_keep:])):
            positional_sequence = []
            index = num_utterances_to_keep - i - 1
            index = torch.tensor(index).to(self.device)
            # print('i=',i,'len=',len(states))

            for state in states:
                positional_sequence.append(torch.cat([state, self.positional_embedder(index)], dim=0))
                #TODO 350
                # print(torch.cat([state, self.positional_embedder(index)], dim=0).size())
            assert len(positional_sequence) == len(utterance), \
                "Expected utterance and state sequence length to be the same, " \
                + "but they were " + str(len(utterance)) \
                + " and " + str(len(positional_sequence))

            if group:
                new_states.append(positional_sequence)
            #TODO False
            else:
                new_states.extend(positional_sequence)
            flat_sequence.extend(utterance)

        return new_states, flat_sequence

    # def set_dropout(self, value):
    #     """ Sets the dropout to a specified value.
    #
    #     Inputs:
    #         value (float): Value to set dropout to.
    #     """
    #     self.dropout = value

    # def build_optim(self):
    #     params_trainer = []
    #     params_bert_trainer = []
    #     for name, param in self.named_parameters():
    #         if param.requires_grad:
    #             if 'model_bert' in name:
    #                 params_bert_trainer.append(param)
    #             else:
    #                 params_trainer.append(param)
    #     self.trainer = torch.optim.Adam(params_trainer, lr=self.args.initial_lr)
    #     if self.args.fine_tune_bert:
    #         self.bert_trainer = torch.optim.Adam(params_bert_trainer, lr=self.args.lr_bert)


    def _initialize_discourse_states(self):
        # torch.Size([150])
        discourse_state = self.initial_discourse_state

        discourse_lstm_states = []
        #TODO ModuleList(
        #   (0): LSTMCell(300, 150)
        # )
        for lstm in self.discourse_lstms:

            hidden_size = lstm.weight_hh.size()[1]
            if lstm.weight_hh.is_cuda:
                h_0 = torch.cuda.FloatTensor(1,hidden_size).fill_(0)
                c_0 = torch.cuda.FloatTensor(1,hidden_size).fill_(0)
            else:
                h_0 = torch.zeros(1,hidden_size)
                c_0 = torch.zeros(1,hidden_size)
            discourse_lstm_states.append((h_0, c_0))

        return discourse_state, discourse_lstm_states


