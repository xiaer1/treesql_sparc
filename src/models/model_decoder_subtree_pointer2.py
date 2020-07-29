# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# -*- coding: utf-8 -*-
"""
# @Time    : 2019/12/09
# @Author  : xiaxia.wang
# @File    : src/models/model.py
# @Software: PyCharm
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
from torch.autograd import Variable

from src.beam import Beams, ActionInfo
from src.models.basic_model_decoder import BasicModel
from src.rule import semQL as define_rule
from src.models.bert import utils_bert
from src.models.encoder import Encoder
from src.models.attention import Attention,AttentionResult
from src.models import nn_utils
from src.models import torch_utils
from src.models.pointer_net import PointerNet

class EditTreeNet(BasicModel):
    def __init__(self,args,grammar):
        super(EditTreeNet,self).__init__(args)
        self.args = args
        self.grammar = grammar
        self.use_bert = args.use_bert

        if args.cuda:
            self.new_long_tensor = torch.cuda.LongTensor
            self.new_tensor = torch.cuda.FloatTensor
        else:
            self.new_long_tensor = torch.LongTensor
            self.new_tensor = torch.FloatTensor

        self.dropout_layer = nn.Dropout(args.dropout)

        self.production_readout_b = nn.Parameter(torch.FloatTensor(len(grammar.prod2id)).zero_())
        self.production_embed = nn.Embedding(len(grammar.prod2id), args.action_embed_size)
        self.type_embed = nn.Embedding(len(grammar.type2id), args.type_embed_size)

        # initial the embedding layers
        nn.init.xavier_normal_(self.production_embed.weight.data)
        nn.init.xavier_normal_(self.type_embed.weight.data)



        self.read_out_act = torch.tanh if args.readout == 'non_linear' else nn_utils.identity

        self.query_vec_to_action_embed = nn.Linear(args.att_vec_size, args.action_embed_size,
                                                   bias=args.readout == 'non_linear')

        self.schema_rnn_input = nn.Linear(args.col_embed_size, args.action_embed_size, bias=False)
        self.turn_rnn_input = nn.Linear(args.col_embed_size, args.action_embed_size, bias=False)
        self.valueS_rnn_input = nn.Linear(args.col_embed_size, args.action_embed_size, bias=False)
        self.distance_rnn_input = nn.Linear(args.col_embed_size, args.action_embed_size, bias=False)

        self.previous_action_embedding_linear = nn.Linear(args.action_embed_size,args.col_embed_size,bias=False)

        # TODO (B,300) -> (B,128)
        # self.read_out_act -> f(x) = x
        # linear(input, weight, bias=None)   weight和bias都是学习参数
        self.production_readout = lambda q: F.linear(self.read_out_act(self.query_vec_to_action_embed(q)),
                                                     self.production_embed.weight, self.production_readout_b)


        if args.use_schema_encoder:
            # Create the schema encoder
            schema_encoder_num_layer = 1
            #TODO 300
            schema_encoder_input_size = args.input_embedding_size
            schema_encoder_state_size = args.encoder_state_size
            if args.use_bert:
                schema_encoder_input_size = self.bert_config.hidden_size #self.bert_config.hidden_size

            self.schema_encoder = Encoder(schema_encoder_num_layer, schema_encoder_input_size, schema_encoder_state_size)

        # self-attention
        if self.args.use_schema_self_attention:
            # TODO schema_attention_key_size : 300
            self.schema2schema_attention_module = Attention(self.schema_attention_key_size,
                                                            self.schema_attention_key_size,
                                                            self.schema_attention_key_size)
        # utterance level attention
        if self.args.use_utterance_attention:
            self.utterance_attention_module = Attention(self.args.encoder_state_size,
                                                        self.args.encoder_state_size,
                                                        self.args.encoder_state_size)

        # Previous query Encoder
        if self.args.use_previous_query:
            self.query_encoder = Encoder(args.encoder_num_layers, args.hidden_size,
                                         args.hidden_size)

            self.query_token_weights = torch_utils.add_params(
                (self.args.encoder_state_size, self.args.encoder_state_size), "weights-query-token")


        '''
        if self.args.use_encoder_attention:
            #TODO :  query key value
            self.utterance2schema_attention_module = Attention(self.schema_attention_key_size, self.utterance_attention_key_size, self.utterance_attention_key_size)
            #TODO attention_key_size = params.encoder_state_size + params.positional_embedding_size : 300 + 50
            self.schema2utterance_attention_module = Attention(self.utterance_attention_key_size, self.schema_attention_key_size, self.schema_attention_key_size)

            new_attention_key_size = self.schema_attention_key_size + self.utterance_attention_key_size
            self.schema_attention_key_size = new_attention_key_size
            self.utterance_attention_key_size = new_attention_key_size
            if self.args.use_schema_encoder_2:
                # TODO : self.schema_attention_key_size = 650
                self.schema_encoder_2 = Encoder(schema_encoder_num_layer, self.schema_attention_key_size, self.schema_attention_key_size)
                # TODO : self.utterance_attention_key_size = 650
                self.utterance_encoder_2 = Encoder(args.encoder_num_layers, self.utterance_attention_key_size,self.utterance_attention_key_size)
        '''

        sketch_input_dim = args.action_embed_size + args.type_embed_size + \
                           self.utterance_attention_key_size + \
                           args.encoder_state_size
        lf_input_dim = args.action_embed_size + args.type_embed_size + \
                           self.utterance_attention_key_size + self.schema_attention_key_size + \
                           args.encoder_state_size

        self.sketch_decoder_lstm = nn.LSTMCell(sketch_input_dim, args.hidden_size)
        self.lf_decoder_lstm = nn.LSTMCell(lf_input_dim, args.hidden_size)

        self.decoder_cell_init = nn.Linear(self.args.hidden_size, args.hidden_size)
        if self.args.use_query_attention:
            self.query_attention_module = Attention(self.args.hidden_size, self.args.hidden_size,
                                                    self.args.hidden_size)

            self.start_query_attention_vector = torch_utils.add_params((self.args.encoder_state_size,),
                                                                       "start_query_attention_vector")

        sketch_att_vec_linear_input = args.hidden_size + self.utterance_attention_key_size
        lf_att_vec_linear_input = args.hidden_size + self.utterance_attention_key_size + self.schema_attention_key_size
        if self.args.use_query_attention:
            sketch_att_vec_linear_input += args.encoder_state_size
            lf_att_vec_linear_input += args.encoder_state_size

        self.sketch_att_vec_linear = nn.Linear(
            sketch_att_vec_linear_input,
            args.att_vec_size, bias=False)
        self.lf_att_vec_linear = nn.Linear(
            lf_att_vec_linear_input,
            args.att_vec_size, bias=False)

        # self.state_transform_weights_sketch = torch_utils.add_params(
        #     (sketch_att_vec_linear_input,
        #      self.args.encoder_state_size), "weights-state-transform_sketch")
        #
        # self.state_transform_weights_lf = torch_utils.add_params(
        #     (lf_att_vec_linear_input,
        #      self.args.encoder_state_size), "weights-state-transform_lf")

        self.schema_att_vec_linear = nn.Linear(
            lf_att_vec_linear_input,
            self.schema_attention_key_size, bias=False)

        self.schema_pointer_net = PointerNet(args.att_vec_size,self.schema_attention_key_size)
        self.query_pointer_net = PointerNet(args.att_vec_size, self.schema_attention_key_size)
        self.turn_pointer_net =  PointerNet(args.att_vec_size, self.schema_attention_key_size)
        self.valueS_pointer_net = PointerNet(args.att_vec_size, self.schema_attention_key_size)
        self.distance_pointer_net = PointerNet(args.att_vec_size, self.schema_attention_key_size)

        self.prob_att = nn.Linear(args.att_vec_size, 1)

        self.attention_module_utter = Attention(self.args.encoder_state_size, self.utterance_attention_key_size,
                                          self.utterance_attention_key_size)
        self.attention_module_schema = Attention(self.args.encoder_state_size, self.schema_attention_key_size,
                                                self.schema_attention_key_size)

        if self.args.use_copy_switch:
            if self.args.use_query_attention:
                #TODO 300 + 650  + 300 ,1
                self.state2copyswitch_transform_weights = torch_utils.add_params((args.att_vec_size, 1), "weights-state-transform_sketch")
                # self.state2copyswitch_transform_weights_lf = torch_utils.add_params(
                #     (lf_att_vec_linear_input, 1),"weights-state-transform_lf")
            else:
                raise ValueError('Now not implement')

    def forward(self,idx,epoch,interaction,optimizer,bert_optimizer):
        '''
        :param interaction:  object
                 Attribute dict_keys(['interaction_list', 'tab_cols', 'col_iter', 'tab_ids', 'index', 'table_names'])
        interaction 和 IRNet中的examples 都是 一个batch，但是该interaction的SQL是有先后顺序的，需要融合editSQL的处理方式
            1. question要像 editSQL一样层层 编码
            2.
        :return:
        '''

        losses = []
        total_gold_tokens = 0

        input_hidden_states = []
        input_sequences = []

        final_utterance_states_c = []
        final_utterance_states_h = []

        previous_lf_query_states = []
        previous_lf_queries = []

        previous_sketch_query_states = []
        previous_sketch_queries = []

        decoder_states = []
        discourse_state = None

        if self.args.discourse_level_lstm:
            discourse_state, discourse_lstm_states = self._initialize_discourse_states()

        utterances = interaction.interaction_list
        # column_names_embedder_input = interaction.column_names_embedder_input
        batch_sketch_loss = []
        batch_lf_loss = []

        sketch_gold_token_cnts =[]
        lf_gold_token_cnts = []

        turn_states_list = []
        # use for locate value turn
        utterance_states_list_store = []
        for utterance_index, utterance in enumerate(utterances):
            # print('#Turn {}#,question:{},sql:{}'.format(utterance_index,utterance.one_utterance,utterance.gold_query))
            #TODO 1. 问句编码
            # input_sequence = utterance.one_utterance
            input_sequence = utterance.one_utterance_linking
            # input_sequence = utterance.union_utterance


            previous_lf,previous_sketch = interaction.previous_query(utterance_index)

            # TODO Encode the utterance, and update the discourse-level states
            if not self.args.use_bert:
                pass
            else:
                '''
                utterance_states : list # torch.Size([7, 300])  print(torch.stack(utterance_states,dim=0).size())
                schema_states : list # torch.Size([42, 300])  print(torch.stack(schema_states,dim=0).size()
                final_utterance_state : tuple (c,h)  c and h are all list[num_layers] , c[-1] : last layer
                '''
                final_utterance_state, utterance_states, schema_states = self.get_bert_encoding(input_sequence,
                                                                                                interaction,
                                                                                                discourse_state,
                                                                                                dropout=True)
                # utterance_states = [torch.randn(300)] * len(input_sequence)
                utterance_states_list_store.append(utterance_states)
                #
                # schema_states = [torch.randn(300)] * len(interaction.column_names_embedder_input)
                # interaction.set_column_name_embeddings(schema_states)
                # final_utterance_state = ([torch.randn(300)], [torch.randn(300)])
                turn_states_list.append(final_utterance_state[1][-1])
                #torch.Size([300]) torch.Size([300])
                # print(final_utterance_state[0][0].size(),final_utterance_state[1][0].size())
                # utterance_states = torch.stack(utterance_states, dim=0).unsqueeze(0)
                # schema_states = torch.stack(schema_states,dim=0).unsqueeze(0)
                #TODO is dropout
                # utterance_states = self.dropout_layer(utterance_states)
                # last_cell = final_utterance_state[0][-1].view(1,-1)
            input_hidden_states.extend(utterance_states)
            input_sequences.append(input_sequence)
            num_utterances_to_keep = min(self.args.maximum_utterances, len(input_sequences))
            if self.args.discourse_level_lstm:
                _, discourse_state, discourse_lstm_states = torch_utils.forward_one_multilayer(self.discourse_lstms,
                                                                                               final_utterance_state[1][0],
                                                                                               discourse_lstm_states,
                                                                                               self.dropout_ratio)
            if self.args.use_utterance_attention:
                final_utterance_states_c, final_utterance_states_h, final_utterance_state = self.get_utterance_attention(
                    final_utterance_states_c, final_utterance_states_h, final_utterance_state, num_utterances_to_keep)

            if self.args.state_positional_embeddings:
                # TODO 给每个question token embedding 300 + 位置embedding 50，得到新的embedding，
                full_utterance_states, flat_sequence = self._add_positional_embeddings(input_hidden_states, input_sequences,group=True)
                utterance_states = full_utterance_states[utterance_index]
            else:
                flat_sequence = []
                for utt in input_sequences[-num_utterances_to_keep:]:
                    flat_sequence.extend(utt)


            if self.args.use_previous_query and len(previous_lf) > 0:
                previous_lf_queries, previous_lf_query_states = self.get_previous_queries(previous_lf_queries,
                                                                                          previous_lf_query_states,
                                                                                          previous_lf,interaction,
                                                                                          turn_states_list, utterance_states_list_store)

            if self.args.use_previous_query and len(previous_sketch) > 0:
                previous_sketch_queries, previous_sketch_query_states = self.get_previous_queries(previous_sketch_queries,
                                                                                          previous_sketch_query_states,
                                                                                          previous_sketch, interaction)
                #TODO [[Root1(3), Root(5), Sel(0), N(1)]]
                # 4 torch.Size([128])
                # print(previous_sketch_queries)
                # print(len(previous_sketch_query_states[0]), previous_sketch_query_states[0][0].size())

            if isinstance(schema_states, list):
                schema_states = torch.stack(schema_states, dim=0).unsqueeze(0)
            if isinstance(utterance_states, list):
                utterance_states = torch.stack(utterance_states, dim=0).unsqueeze(0)
            if isinstance(turn_states_list, list):
                turn_states = torch.stack(turn_states_list, dim=0).unsqueeze(0)

            
            #schema_states,utterance_states = self.get_utterance_schema_attention(schema_states,utterance_states)



            sketch_prob_var,sketch_gold_token_cnt = self.train_sketch_one_turn(utterance,utterance_states,final_utterance_state,previous_sketch_queries,previous_sketch_query_states)

            lf_prob_var,lf_gold_token_cnt = self.train_action_one_turn(utterance,utterance_states,final_utterance_state,
                                                                       interaction,
                                                                       schema_states, turn_states,
                                                                       utterance_states_list_store,
                                                                       previous_lf_queries,previous_lf_query_states)

            batch_sketch_loss.append(sketch_prob_var)
            sketch_gold_token_cnts.append(sketch_gold_token_cnt)

            batch_lf_loss.append(lf_prob_var)
            lf_gold_token_cnts.append(lf_gold_token_cnt)

        interaction.init_schema_appear_mask()

        if batch_sketch_loss:
            cnt_sktech = np.sum(np.array(sketch_gold_token_cnts))
            loss_sketch = torch.sum(torch.stack(batch_sketch_loss,dim=0))
            loss_sketch = loss_sketch / cnt_sktech
        if batch_lf_loss:
            cnt_lf = np.sum(np.array(lf_gold_token_cnts))
            loss_lf = torch.sum(torch.stack(batch_lf_loss,dim=0))
            loss_lf = loss_lf / cnt_lf

        if epoch > self.args.loss_epoch_threshold:   #TODO loss_epoch_threshold = 50 ， sketch_loss_coefficient = 1
            loss = loss_lf + self.args.sketch_loss_coefficient * loss_sketch
        else:
            loss = loss_lf + loss_sketch


        loss.backward()

        if self.args.clip_grad > 0.:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip_grad)

        optimizer.step()
        if self.args.fine_tune_bert:
            bert_optimizer.step()
        self.zero_grad()

        loss_scalar = loss.item()
        sk = loss_sketch.item()
        lf = loss_lf.item()

        return loss_scalar,sk,lf
    def get_utterance_schema_attention(self,schema_states,utterance_states):
        # TODO ==========加上 utterance <-> schema attention ==============
        if self.args.use_encoder_attention:

            # torch.Size([350, 28])
            schema_attention = self.utterance2schema_attention_module(torch.stack(schema_states, dim=0), utterance_states).vector
            # torch.Size([300, 6])
            utterance_attention = self.schema2utterance_attention_module(torch.stack(utterance_states, dim=0),schema_states).vector
            if schema_attention.dim() == 1:
                schema_attention = schema_attention.unsqueeze(1)
            if utterance_attention.dim() == 1:
                utterance_attention = utterance_attention.unsqueeze(1)
            # (schema_encoder_size + utterance_encoder_size , len(schema))
            new_schema_states = torch.cat([torch.stack(schema_states, dim=1), schema_attention], dim=0)
            schema_states = list(torch.split(new_schema_states, split_size_or_sections=1, dim=1))
            schema_states = [schema_state.squeeze() for schema_state in schema_states]

            #  # (input_value_size+schema_value_size) x len(input)
            new_utterance_states = torch.cat([torch.stack(utterance_states, dim=1), utterance_attention], dim=0)
            utterance_states = list(torch.split(new_utterance_states, split_size_or_sections=1, dim=1))
            # TODO input_hidden_state 是650维
            utterance_states = [utterance_state.squeeze() for utterance_state in utterance_states]

            # bi-lstm over schema_states and input_hidden_states (embedder is an identify function)
            if self.args.use_schema_encoder_2:
                final_schema_state, schema_states = self.schema_encoder_2(schema_states, lambda x: x,dropout_amount=self.dropout_ratio)
                final_utterance_state, utterance_states = self.utterance_encoder_2(utterance_states,lambda x: x,dropout_amount=self.dropout_ratio)

        if isinstance(schema_states, list):
            schema_states = torch.stack(schema_states, dim=0).unsqueeze(0)
        if isinstance(utterance_states, list):
            utterance_states = torch.stack(utterance_states, dim=0).unsqueeze(0)

        return schema_states,utterance_states

    def get_previous_queries(self, previous_queries, previous_query_states, previous_query,interaction,turn_states_list = None, utterance_states_list = None):
        previous_queries.append(previous_query)

        num_queries_to_keep = min(self.args.maximum_queries,len(previous_queries))

        previous_queries = previous_queries[-num_queries_to_keep:]


        query_token_embedder = lambda query_token,turn_idx,valueS_idx: self.get_query_token_embedding(query_token,interaction, turn_states_list,
                                                                                                      utterance_states_list,turn_idx,valueS_idx)
        _, previous_outputs = self.query_encoder(previous_query, query_token_embedder, dropout_amount=self.dropout_ratio)

        assert len(previous_outputs) == len(previous_query)
        previous_query_states.append(previous_outputs)
        previous_query_states = previous_query_states[-num_queries_to_keep:]

        return previous_queries,previous_query_states

    def get_query_token_embedding(self, output_token, interaction, turn_states_list = None,
                                  utterance_states_list = None, turn_idx = None, valueS_idx = None):
        '''
        :param output_token:
        :param interaction:
        :return:
        '''
        if type(output_token) in [define_rule.Root1,
                                    define_rule.Root,
                                    define_rule.Sel,
                                    define_rule.Filter,
                                    define_rule.Sup,
                                    define_rule.N,
                                    define_rule.Order,
                                    define_rule.A]:

            output_token_embedding = self.previous_action_embedding_linear(self.production_embed.weight[self.grammar.prod2id[output_token.production]])
        elif type(output_token) == define_rule.TC:

            output_token_embedding = interaction.column_name_embeddings[output_token.id_c]
        elif type(output_token) == define_rule.Turn:

            output_token_embedding = turn_states_list[output_token.id_c]

        elif type(output_token) == define_rule.ValueS:

            output_token_embedding = utterance_states_list[turn_idx][output_token.id_c]

        elif type(output_token) == define_rule.Distance:

            output_token_embedding = utterance_states_list[turn_idx][valueS_idx + output_token.id_c]
        else:
            raise ValueError('not implement error!!!')

        return output_token_embedding

    def predict_with_gold_queries(self,interaction,beam_size=1,use_gold_query=True):

        discourse_state = None
        if self.args.discourse_level_lstm:
            discourse_state, discourse_lstm_states = self._initialize_discourse_states()
        utterances = interaction.interaction_list

        input_hidden_states = []
        input_sequences = []

        final_utterance_states_c = []
        final_utterance_states_h = []

        interaction_lf_actions = []
        interaction_sketch_actions = []

        previous_sketch_query_states = []
        previous_sketch_queries = []

        previous_lf_query_states = []
        previous_lf_queries = []

        for utterance_index, utterance in enumerate(utterances):

            #input_sequence = utterance.one_utterance
            input_sequence = utterance.one_utterance_linking
            # input_sequence = utterance.union_utterance

            previous_lf,previous_sketch = interaction.previous_query(utterance_index)

            # TODO Encode the utterance, and update the discourse-level states
            if not self.args.use_bert:
                raise ValueError('Should use bert!!!')
            else:
                '''
                utterance_states : list # torch.Size([7, 300])  print(torch.stack(utterance_states,dim=0).size())
                schema_states : list # torch.Size([42, 300])  print(torch.stack(schema_states,dim=0).size()
                final_utterance_state : tuple (c,h)  c and h are all list[num_layers] , c[-1] : last layer
                '''
                final_utterance_state, utterance_states, schema_states = self.get_bert_encoding(input_sequence,
                                                                                                interaction,
                                                                                                discourse_state,
                                                                                                dropout=True)

            input_hidden_states.extend(utterance_states)
            input_sequences.append(input_sequence)
            num_utterances_to_keep = min(self.args.maximum_utterances, len(input_sequences))
            if self.args.discourse_level_lstm:
                _, discourse_state, discourse_lstm_states = torch_utils.forward_one_multilayer(self.discourse_lstms,
                                                                                               final_utterance_state[1][0],
                                                                                               discourse_lstm_states,
                                                                                               self.dropout_ratio)
            if self.args.use_utterance_attention:
                final_utterance_states_c, final_utterance_states_h, final_utterance_state = self.get_utterance_attention(
                    final_utterance_states_c, final_utterance_states_h, final_utterance_state, num_utterances_to_keep)

            if self.args.state_positional_embeddings:
                # TODO 给每个question token embedding 300 + 位置embedding 50，得到新的embedding，
                full_utterance_states, flat_sequence = self._add_positional_embeddings(input_hidden_states,
                                                                                       input_sequences, group=True)
            else:
                flat_sequence = []
                for utt in input_sequences[-num_utterances_to_keep:]:
                    flat_sequence.extend(utt)

            #TODO use previous query
            if self.args.use_previous_query and len(previous_sketch) > 0:
                previous_sketch_queries, previous_sketch_query_states = self.get_previous_queries(previous_sketch_queries,
                                                                                                  previous_sketch_query_states,
                                                                                                  previous_sketch,
                                                                                                  interaction)

            if self.args.use_previous_query and len(previous_lf) > 0:
                previous_lf_queries, previous_lf_query_states = self.get_previous_queries(previous_lf_queries,
                                                                                          previous_lf_query_states,
                                                                                          previous_lf, interaction)
            utterance_states = full_utterance_states[utterance_index]

            if isinstance(schema_states, list):
                schema_states = torch.stack(schema_states, dim=0).unsqueeze(0)
            if isinstance(utterance_states, list):
                utterance_states = torch.stack(utterance_states, dim=0).unsqueeze(0)
            #schema_states, utterance_states = self.get_utterance_schema_attention(schema_states, utterance_states)


            completed_sketch_beams = self.predict_sketch_one_turn(beam_size,utterance_states,final_utterance_state,previous_sketch_queries,previous_sketch_query_states)


            if completed_sketch_beams == []:
                sketch_actions = []
                lf_actions = []
            else:
                sketch_actions = completed_sketch_beams[0].actions
                padding_sketch = self.padding_sketch(sketch_actions)

                completed_lf_beams = self.predict_action_one_turn(beam_size,interaction,padding_sketch,utterance_states,final_utterance_state,
                                                                  schema_states,previous_lf_queries,previous_lf_query_states)
                if completed_lf_beams == []:
                    lf_actions = []
                else:
                    lf_actions = completed_lf_beams[0].actions

            interaction_sketch_actions.append(sketch_actions)
            interaction_lf_actions.append(lf_actions)

        interaction.init_schema_appear_mask()

        return interaction_lf_actions,interaction_sketch_actions



    def predict_with_predicted_queries(self,interaction,beam_size=1):
        discourse_state = None
        if self.args.discourse_level_lstm:
            discourse_state, discourse_lstm_states = self._initialize_discourse_states()
        utterances = interaction.interaction_list

        input_hidden_states = []
        input_sequences = []

        final_utterance_states_c = []
        final_utterance_states_h = []

        interaction_lf_actions = []
        interaction_sketch_actions = []

        previous_sketch_query_states = []
        previous_sketch_queries = []

        previous_lf_query_states = []
        previous_lf_queries = []

        utterance_states_list = []
        turn_states_list = []
        for utterance_index, utterance in enumerate(utterances):

            #input_sequence = utterance.one_utterance
            input_sequence = utterance.one_utterance_linking
            # input_sequence2 = utterance.union_utterance


            previous_predicted_lf, previous_predicted_sketch = interaction.previous_predicted_query(utterance_index)


            # TODO Encode the utterance, and update the discourse-level states
            if not self.args.use_bert:
                raise ValueError('Should use bert!!!')
            else:
                '''
                utterance_states : list # torch.Size([7, 300])  print(torch.stack(utterance_states,dim=0).size())
                schema_states : list # torch.Size([42, 300])  print(torch.stack(schema_states,dim=0).size()
                final_utterance_state : tuple (c,h)  c and h are all list[num_layers] , c[-1] : last layer
                '''
                final_utterance_state, utterance_states, schema_states = self.get_bert_encoding(input_sequence,
                                                                                                interaction,
                                                                                                discourse_state,
                                                                                                dropout=True)
                # utterance_states = [torch.randn(300)] * len(input_sequence)
                utterance_states_list.append(utterance_states)
                #
                # schema_states = [torch.randn(300)] * len(interaction.column_names_embedder_input)
                # interaction.set_column_name_embeddings(schema_states)
                # final_utterance_state = ([torch.randn(300)], [torch.randn(300)])
                turn_states_list.append(final_utterance_state[1][-1])


            input_hidden_states.extend(utterance_states)
            input_sequences.append(input_sequence)
            num_utterances_to_keep = min(self.args.maximum_utterances, len(input_sequences))
            if self.args.discourse_level_lstm:
                _, discourse_state, discourse_lstm_states = torch_utils.forward_one_multilayer(self.discourse_lstms,
                                                                                               final_utterance_state[1][0],
                                                                                               discourse_lstm_states,
                                                                                               self.dropout_ratio)
            if self.args.use_utterance_attention:
                final_utterance_states_c, final_utterance_states_h, final_utterance_state = self.get_utterance_attention(
                    final_utterance_states_c, final_utterance_states_h, final_utterance_state, num_utterances_to_keep)

            if self.args.state_positional_embeddings:
                # TODO 给每个question token embedding 300 + 位置embedding 50，得到新的embedding，
                full_utterance_states, flat_sequence = self._add_positional_embeddings(input_hidden_states,
                                                                                       input_sequences, group=True)
            else:
                flat_sequence = []
                for utt in input_sequences[-num_utterances_to_keep:]:
                    flat_sequence.extend(utt)

            # TODO use previous query
            if self.args.use_previous_query and len(previous_predicted_sketch) > 0:
                previous_sketch_queries, previous_sketch_query_states = self.get_previous_queries(
                    previous_sketch_queries,
                    previous_sketch_query_states,
                    previous_predicted_sketch,
                    interaction)

            if self.args.use_previous_query and len(previous_predicted_lf) > 0:
                previous_lf_queries, previous_lf_query_states = self.get_previous_queries(previous_lf_queries,
                                                                                          previous_lf_query_states,
                                                                                          previous_predicted_lf, interaction,
                                                                                          turn_states_list, utterance_states_list)
            utterance_states = full_utterance_states[utterance_index]


            if isinstance(schema_states, list):
                schema_states = torch.stack(schema_states, dim=0).unsqueeze(0)
            if isinstance(utterance_states, list):
                utterance_states = torch.stack(utterance_states, dim=0).unsqueeze(0)
            if isinstance(turn_states_list, list):
                turn_states = torch.stack(turn_states_list, dim=0).unsqueeze(0)

            #schema_states, utterance_states = self.get_utterance_schema_attention(schema_states, utterance_states)

            completed_sketch_beams = self.predict_sketch_one_turn(beam_size, utterance_states, final_utterance_state,
                                                                  previous_sketch_queries, previous_sketch_query_states)

            if completed_sketch_beams == []:
                sketch_actions = []
                lf_actions = []
            else:
                sketch_actions = completed_sketch_beams[0].actions
                padding_sketch = self.padding_sketch(sketch_actions)

                completed_lf_beams = self.predict_action_one_turn(beam_size, interaction, padding_sketch,
                                                                  utterance_states, final_utterance_state,
                                                                  schema_states, turn_states, utterance_states_list,
                                                                  previous_lf_queries,
                                                                  previous_lf_query_states)
                if completed_lf_beams == []:
                    lf_actions = []
                else:
                    lf_actions = completed_lf_beams[0].actions

            interaction.predicted_sketch_action.append(sketch_actions)
            interaction.predicted_lf_action.append(lf_actions)
            interaction_sketch_actions.append(sketch_actions)
            interaction_lf_actions.append(lf_actions)

        interaction.predicted_sketch_action = []
        interaction.predicted_lf_action = []

        interaction.init_schema_appear_mask()

        return interaction_lf_actions, interaction_sketch_actions

    def predict_sketch_one_turn(self,beam_size,utterance_states,final_utterance_state,previous_sketch_queries,previous_sketch_query_states):
        t = 0
        beams = [Beams(is_sketch=True)]
        completed_beams = []

        last_cell = final_utterance_state[0][-1].unsqueeze(0)
        dec_init_vec = self.init_decoder_state(last_cell)
        h_tm1 = dec_init_vec

        while (len(completed_beams)) < beam_size and  t < self.args.decode_max_time_step:
            # hyp_num = len(beams)

            if t == 0:
                x = self.new_tensor(1, self.sketch_decoder_lstm.input_size).zero_()
            else:
                # TODO action
                a_tm1_embeds = []
                for e_id, hyp in enumerate(beams):
                    action_tm1 = hyp.actions[-1]
                    if type(action_tm1) in [define_rule.Root1,
                                            define_rule.Root,
                                            define_rule.Sel,
                                            define_rule.Filter,
                                            define_rule.Sup,
                                            define_rule.N,
                                            define_rule.Order]:
                        a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                    else:
                        raise ValueError('unknown action %s' % action_tm1)

                    a_tm1_embeds.append(a_tm1_embed)
                a_tm1_embeds = torch.stack(a_tm1_embeds)

                inputs = [a_tm1_embeds]

                # TODO action type
                pre_types = []
                for e_id, hyp in enumerate(beams):
                    action_tm = hyp.actions[-1]

                    pre_type = self.type_embed.weight[self.grammar.type2id[type(action_tm)]]
                    pre_types.append(pre_type)

                pre_types = torch.stack(pre_types)

                inputs.append(pre_types)
                inputs.append(att_tm1)

                x = torch.cat(inputs, dim=-1)

            (h_t, cell_t) ,state_and_attn,att = self.decoder_sketch_one_step_query(x, h_tm1, utterance_states,
                                                         self.sketch_decoder_lstm,previous_sketch_query_states)

            att_t = torch.tanh(self.sketch_att_vec_linear(state_and_attn))
            att_t = self.dropout_layer(att_t)

            # TODO torch.Size([1, 46])
            if att_t.dim() == 1:
                att_t = att_t.unsqueeze(0)

            apply_rule_log_prob = torch.log_softmax(self.production_readout(att_t), dim=-1)

            # TODO use previous sketck query
            # intermediate_state = self._get_intermediate_state_sketch(state_and_attn, dropout_amount=self.dropout_ratio)
            if self.args.use_previous_query and len(previous_sketch_queries) > 0:
                if self.args.use_copy_switch:
                    # copy_switch = self._get_copy_switch_sketch(state_and_attn)
                    copy_switch = self._get_copy_switch(att_t)

                assert len(previous_sketch_queries) == 1, 'debug mode'

                previous_sketch_query_state = previous_sketch_query_states[-1]

                previous_sketch_query_state = torch.stack(previous_sketch_query_state, dim=0).repeat(1, 1, 1)
                # [B, len_query]
                query_scores = self.query_pointer_net(previous_sketch_query_state, att_t.unsqueeze(0))
                query_scores = torch.log_softmax(query_scores, dim=-1).squeeze(0)

                query_scores = query_scores * copy_switch
                apply_rule_log_prob = apply_rule_log_prob * (1 - copy_switch)

            if att.dim() == 1:
                att = att.unsqueeze(0)

            new_hyp_meta = []

            for hyp_id, hyp in enumerate(beams):

                action_class = hyp.get_availableClass()

                if action_class in [define_rule.Root1,
                                    define_rule.Root,
                                    define_rule.Sel,
                                    define_rule.Filter,
                                    define_rule.Sup,
                                    define_rule.N,
                                    define_rule.Order]:
                    if action_class in [define_rule.Root]:
                        parent_type = define_rule.action_map[define_rule.Root1]
                    elif action_class in [define_rule.Sel,define_rule.Filter,define_rule.Sup,define_rule.Order]:
                        parent_type = define_rule.action_map[define_rule.Root]
                    elif action_class in [define_rule.N]:
                        parent_type = define_rule.action_map[define_rule.Sel]
                    elif action_class in [define_rule.Root1]:
                        parent_type = None
                    possible_productions = self.grammar.get_production(action_class)
                    # print(action_class,possible_productions)
                    for possible_production in possible_productions:
                        prod_id = self.grammar.prod2id[possible_production]
                        prod_score = apply_rule_log_prob[hyp_id, prod_id]

                        query_scores_list = []
                        if self.args.use_previous_query and len(previous_sketch_queries) > 0:

                            for i,item in enumerate(previous_sketch_queries[-1]):
                                if item.production == possible_production and item.parent == parent_type:
                                    query_scores_list.append(query_scores[i])
                        if query_scores_list:
                            prod_score += torch.stack(query_scores_list,dim=0).sum()

                        new_hyp_score = hyp.score + prod_score.data.cpu()
                        meta_entry = {'action_type': action_class, 'prod_id': prod_id,
                                      'score': prod_score, 'new_hyp_score': new_hyp_score,
                                      'prev_hyp_id': hyp_id}
                        new_hyp_meta.append(meta_entry)
                else:
                    print(action_class)
                    raise RuntimeError('No right action class')
            if not new_hyp_meta:
                break

            new_hyp_scores = torch.stack([x['new_hyp_score'] for x in new_hyp_meta], dim=0)

            top_new_hyp_scores, meta_ids = torch.topk(new_hyp_scores,
                                                      k=min(new_hyp_scores.size(0), beam_size - len(completed_beams)))

            live_hyp_ids = []
            new_beams = []

            for new_hyp_score, meta_id in zip(top_new_hyp_scores.data.cpu(), meta_ids.data.cpu()):
                action_info = ActionInfo()
                hyp_meta_entry = new_hyp_meta[meta_id]
                prev_hyp_id = hyp_meta_entry['prev_hyp_id']
                prev_hyp = beams[prev_hyp_id]
                action_type_str = hyp_meta_entry['action_type']
                prod_id = hyp_meta_entry['prod_id']

                assert prod_id < len(self.grammar.id2prod), 'prod_id must be in grammar.id2prod'

                production = self.grammar.id2prod[prod_id]
                
                # ( id_c , parent)
                if action_type_str in [define_rule.Root]:
                    parent_type = define_rule.action_map[define_rule.Root1]
                elif action_type_str in [define_rule.Sel,define_rule.Filter,define_rule.Sup,define_rule.Order]:
                    parent_type = define_rule.action_map[define_rule.Root]
                elif action_type_str in [define_rule.N]:
                    parent_type = define_rule.action_map[define_rule.Sel]
                elif action_type_str in [define_rule.Root1]:
                    parent_type = None    
                action = action_type_str(list(action_type_str._init_grammar()).index(production),parent_type)

                action_info.action = action
                action_info.t = t
                action_info.score = hyp_meta_entry['score']

                new_hyp = prev_hyp.clone_and_apply_action_info(action_info)

                new_hyp.score = new_hyp_score
                new_hyp.inputs.extend(prev_hyp.inputs)

                if new_hyp.is_valid is False:
                    continue

                if new_hyp.completed:
                    completed_beams.append(new_hyp)
                else:
                    new_beams.append(new_hyp)
                    live_hyp_ids.append(prev_hyp_id)

            #print('Select action',action)

            if live_hyp_ids:
                h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
                att_tm1 = att[live_hyp_ids]
                beams = new_beams
                t += 1
            else:
                break

        if len(completed_beams) == 0:
            print('Predict sketch failed!!!\n')
            return []

        completed_beams.sort(key=lambda hyp: -hyp.score)
        return completed_beams


    def predict_action_one_turn(self,beam_size,interaction,padding_sketch,utterance_states,final_utterance_state,
                                schema_states, turn_states, utterance_states_list_store,
                                previous_lf_queries,previous_lf_query_states):

        last_cell = final_utterance_state[0][-1].unsqueeze(0)
        dec_init_vec = self.init_decoder_state(last_cell)
        h_tm1 = dec_init_vec

        t = 0
        beams = [Beams(is_sketch=False)]
        completed_beams = []
        assert beam_size == 1
        turn_enable = np.zeros(shape=beam_size, dtype=np.int32)
        valueS_enable = np.zeros(shape=beam_size, dtype=np.int32)

        while len(completed_beams) < beam_size and t < self.args.decode_max_time_step:
            if t == 0:
                x = self.new_tensor(1, self.lf_decoder_lstm.input_size).zero_()
            else:
                a_tm1_embeds = []
                pre_types = []

                for e_id, hyp in enumerate(beams):
                    action_tm1 = hyp.actions[-1]
                    if type(action_tm1) in [define_rule.Root1,
                                            define_rule.Root,
                                            define_rule.Sel,
                                            define_rule.Filter,
                                            define_rule.Sup,
                                            define_rule.N,
                                            define_rule.Order]:

                        a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                        hyp.sketch_step += 1
                    elif isinstance(action_tm1, define_rule.TC):
                        a_tm1_embed = self.schema_rnn_input(interaction.column_name_embeddings[action_tm1.id_c])
                    elif isinstance(action_tm1, define_rule.A):
                        a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                    elif isinstance(action_tm1, define_rule.Turn):
                        a_tm1_embed = self.turn_rnn_input(turn_states[0][action_tm1.id_c])
                    elif isinstance(action_tm1, define_rule.ValueS):
                        a_tm1_embed = self.valueS_rnn_input(utterance_states_list_store[turn_enable[0]][action_tm1.id_c])
                    elif isinstance(action_tm1, define_rule.Distance):
                        a_tm1_embed = self.distance_rnn_input(
                            utterance_states_list_store[turn_enable[0]][valueS_enable[0] + action_tm1.id_c])
                    else:
                        raise ValueError('unknown action %s' % action_tm1)

                    a_tm1_embeds.append(a_tm1_embed)

                a_tm1_embeds = torch.stack(a_tm1_embeds)

                inputs = [a_tm1_embeds]

                for e_id, hyp in enumerate(beams):
                    action_tm = hyp.actions[-1]
                    pre_type = self.type_embed.weight[self.grammar.type2id[type(action_tm)]]
                    pre_types.append(pre_type)

                pre_types = torch.stack(pre_types)

                inputs.append(pre_types)
                inputs.append(att_tm1)

                x = torch.cat(inputs, dim=-1)


            (h_t, cell_t),state_and_attn,att = self.decoder_lf_one_step_query(x, h_tm1, utterance_states,schema_states,
                                                         self.lf_decoder_lstm,previous_lf_query_states)

            att_t = torch.tanh(self.lf_att_vec_linear(state_and_attn))
            att_t = self.dropout_layer(att_t)
            
            if att_t.dim() == 1:
                att_t = att_t.unsqueeze(0)
            apply_rule_log_prob = torch.log_softmax(self.production_readout(att_t), dim=-1)

            schema_appear_mask = interaction.schema_appear_mask
            schema_appear_mask = torch.from_numpy(schema_appear_mask)  # [B, len]
            if schema_states.is_cuda:
                schema_appear_mask = schema_appear_mask.cuda()

            if self.args.use_column_pointer:
                gate = torch.sigmoid(self.prob_att(att_t))  # [B,1]
                # [B,len_schema]
                schema_weight = self.schema_pointer_net(schema_states, att_t.unsqueeze(0)) * schema_appear_mask * gate + \
                                self.schema_pointer_net(schema_states, att_t.unsqueeze(0)) * (1 - schema_appear_mask) * (1 - gate)
            else:
                schema_weight = self.schema_pointer_net(schema_states, att_t.unsqueeze(0))

            schema_weight = torch.log_softmax(schema_weight, dim=-1)

            turn_weight = self.turn_pointer_net(turn_states, att_t.unsqueeze(0))
            turn_weight = torch.log_softmax(turn_weight, dim=-1)

            #TODO valueS weight
            utter_state = utterance_states_list_store[turn_enable[0]]
            single_states = torch.stack(utter_state, dim=0).unsqueeze(0)

            valueS_weight = self.valueS_pointer_net(single_states, att_t.unsqueeze(0))
            valueS_weight = torch.log_softmax(valueS_weight, dim=-1)


            if att_t.dim() == 1:
                att_t = att_t.unsqueeze(0)

            # TODO use previous sketck query
            # intermediate_state = self._get_intermediate_state_lf(state_and_attn, dropout_amount=self.dropout_ratio)
            if self.args.use_previous_query and len(previous_lf_queries) > 0:
                if self.args.use_copy_switch:
                    copy_switch = self._get_copy_switch(att_t)

                assert len(previous_lf_queries) == 1, 'debug mode'
                previous_lf_query_state = previous_lf_query_states[-1]

                previous_lf_query_state = torch.stack(previous_lf_query_state, dim=0).repeat(1, 1, 1)
                # [B, len_query]
                query_scores = self.query_pointer_net(previous_lf_query_state, att_t.unsqueeze(0))

                query_scores = torch.log_softmax(query_scores, dim=-1).squeeze(0)

                query_scores = query_scores * copy_switch
                apply_rule_log_prob = apply_rule_log_prob * (1 - copy_switch)
                schema_weight = schema_weight * (1 - copy_switch)
                turn_weight = turn_weight * (1 - copy_switch)
                valueS_weight = valueS_weight * (1 - copy_switch)



            if att.dim() == 1:
                att = att.unsqueeze(0)

            new_hyp_meta = []
            assert len(beams) == 1,'debug mode : beams'

            hyp_id = 0
            hyp = beams[hyp_id]
            # for hyp_id, hyp in enumerate(beams):
            if type(padding_sketch[t]) == define_rule.A:
                possible_productions = self.grammar.get_production(define_rule.A)
                for possible_production in possible_productions:
                    prod_id = self.grammar.prod2id[possible_production]
                    prod_score = apply_rule_log_prob[hyp_id, prod_id]

                    query_scores_list = []
                    if self.args.use_previous_query and len(previous_lf_queries) > 0:

                        for i, item in enumerate(previous_lf_queries[-1]):

                            if item.production == possible_production and item.parent == padding_sketch[t].parent:
                                query_scores_list.append(query_scores[i])
                    if query_scores_list:
                        prod_score += torch.stack(query_scores_list, dim=0).sum()

                    new_hyp_score = hyp.score + prod_score.data.cpu()
                    meta_entry = {'action_type': define_rule.A, 'prod_id': prod_id,
                                  'score': prod_score, 'new_hyp_score': new_hyp_score,
                                  'prev_hyp_id': hyp_id}
                    new_hyp_meta.append(meta_entry)
            elif type(padding_sketch[t]) == define_rule.TC:
                for tc_id, _ in enumerate(interaction.column_names_embedder_input):
                    tc_score = schema_weight[hyp_id, tc_id]

                    query_scores_list = []
                    if self.args.use_previous_query and len(previous_lf_queries) > 0:

                        for i, item in enumerate(previous_lf_queries[-1]):
                            if type(item) == define_rule.TC and item.id_c == tc_id and item.parent == padding_sketch[t].parent: 
                                query_scores_list.append(query_scores[i])

                    if query_scores_list:
                        tc_score += torch.stack(query_scores_list, dim=0).sum()


                    new_hyp_score = hyp.score + tc_score.data.cpu()
                    meta_entry = {'action_type': define_rule.TC, 'tc_id': tc_id,
                                  'score': tc_score, 'new_hyp_score': new_hyp_score,
                                  'prev_hyp_id': hyp_id}
                    new_hyp_meta.append(meta_entry)

            elif type(padding_sketch[t]) == define_rule.Turn:
                for turn_id in range(turn_weight.size(1)):
                    turn_score = turn_weight[hyp_id, turn_id]

                    query_scores_list = []
                    if self.args.use_previous_query and len(previous_lf_queries) > 0:

                        for i, item in enumerate(previous_lf_queries[-1]):
                            if type(item) == define_rule.Turn and \
                                    item.id_c == turn_id and \
                                    item.parent == padding_sketch[t].parent:
                                query_scores_list.append(query_scores[i])

                    if query_scores_list:
                        turn_score += torch.stack(query_scores_list, dim=0).sum()

                    new_hyp_score = hyp.score + turn_score.data.cpu()
                    meta_entry = {'action_type': define_rule.Turn, 'turn_id': turn_id,
                                  'score': turn_score, 'new_hyp_score': new_hyp_score,
                                  'prev_hyp_id': hyp_id}
                    new_hyp_meta.append(meta_entry)

            elif type(padding_sketch[t]) == define_rule.ValueS:
                utt_len = len(utterance_states_list_store[turn_enable[0]])
                assert utt_len == valueS_weight.size(1)
                for vs_id in range(valueS_weight.size(1)):
                    vs_score = valueS_weight[hyp_id, vs_id]

                    query_scores_list = []
                    if self.args.use_previous_query and len(previous_lf_queries) > 0:

                        for i, item in enumerate(previous_lf_queries[-1]):
                            if type(item) == define_rule.ValueS and \
                                    item.id_c == vs_id and \
                                    item.parent == padding_sketch[t].parent:
                                query_scores_list.append(query_scores[i])

                    if query_scores_list:
                        vs_score += torch.stack(query_scores_list, dim=0).sum()


                    new_hyp_score = hyp.score + vs_score.data.cpu()
                    meta_entry = {'action_type': define_rule.ValueS, 'vs_id': vs_id,
                                  'score': vs_score, 'new_hyp_score': new_hyp_score,
                                  'prev_hyp_id': hyp_id}
                    new_hyp_meta.append(meta_entry)

            elif type(padding_sketch[t]) == define_rule.Distance:

                # TODO Distance weight
                start_loc = valueS_enable[0]
                end_loc = min(start_loc + self.args.value_max_len, len(utter_state))
                # print('-->',start_loc, len(utter_state))
                dist_states = torch.stack(utter_state[start_loc: end_loc], dim=0).unsqueeze(0)
                dist_weight = self.distance_pointer_net(dist_states, att_t.unsqueeze(0))
                dist_weight = torch.log_softmax(dist_weight, dim=-1)
                if self.args.use_previous_query and len(previous_lf_queries) > 0:
                    if self.args.use_copy_switch:
                        dist_weight = dist_weight * (1 - copy_switch)

                for dist_id in range(dist_weight.size(1)):
                    dist_score = dist_weight[hyp_id, dist_id]

                    query_scores_list = []
                    if self.args.use_previous_query and len(previous_lf_queries) > 0:

                        for i, item in enumerate(previous_lf_queries[-1]):
                            if type(item) == define_rule.Distance and \
                                    item.id_c == dist_id and \
                                    item.parent == padding_sketch[t].parent:
                                query_scores_list.append(query_scores[i])

                    if query_scores_list:
                        dist_score += torch.stack(query_scores_list, dim=0).sum()


                    new_hyp_score = hyp.score + dist_score.data.cpu()
                    meta_entry = {'action_type': define_rule.Distance, 'dist_id': dist_id,
                                  'score': dist_score, 'new_hyp_score': new_hyp_score,
                                  'prev_hyp_id': hyp_id}
                    new_hyp_meta.append(meta_entry)
            else:
                prod_id = self.grammar.prod2id[padding_sketch[t].production]
                new_hyp_score = hyp.score + torch.tensor(0.0)

                meta_entry = {'action_type': type(padding_sketch[t]), 'prod_id': prod_id,
                              'score': torch.tensor(0.0), 'new_hyp_score': new_hyp_score,
                              'prev_hyp_id': hyp_id}

                new_hyp_meta.append(meta_entry)
            if not new_hyp_meta:
                break

            new_hyp_scores = torch.stack([x['new_hyp_score'] for x in new_hyp_meta], dim=0)

            top_new_hyp_scores, meta_ids = torch.topk(new_hyp_scores,
                                                      k=min(new_hyp_scores.size(0),
                                                            beam_size - len(completed_beams)))
            live_hyp_ids = []
            new_beams = []

            for new_hyp_score, meta_id in zip(top_new_hyp_scores.data.cpu(), meta_ids.data.cpu()):
                # print(new_hyp_score,meta_id)
                action_info = ActionInfo()
                hyp_meta_entry = new_hyp_meta[meta_id]
                prev_hyp_id = hyp_meta_entry['prev_hyp_id']
                prev_hyp = beams[prev_hyp_id]

                action_type_str = hyp_meta_entry['action_type']
                if 'prod_id' in hyp_meta_entry:
                    prod_id = hyp_meta_entry['prod_id']
                if action_type_str == define_rule.TC:
                    tc_id = hyp_meta_entry['tc_id']
                    interaction.update_schema_appear_mask(tc_id)
                    action = define_rule.TC(tc_id,padding_sketch[t].parent)

                elif action_type_str == define_rule.Turn:
                    turn_id = hyp_meta_entry['turn_id']
                    turn_enable[0] = turn_id

                    action = define_rule.Turn(turn_id,padding_sketch[t].parent)

                elif action_type_str == define_rule.ValueS:
                    vs_id = hyp_meta_entry['vs_id']
                    valueS_enable[0] = vs_id
                    action = define_rule.ValueS(vs_id,padding_sketch[t].parent)

                elif action_type_str == define_rule.Distance:
                    dist_id = hyp_meta_entry['dist_id']
                    action = define_rule.Distance(dist_id,padding_sketch[t].parent)

                elif prod_id < len(self.grammar.id2prod):
                    production = self.grammar.id2prod[prod_id]
                    action = action_type_str(list(action_type_str._init_grammar()).index(production),padding_sketch[t].parent)
                else:
                    raise NotImplementedError

                action_info.action = action
                action_info.t = t
                action_info.score = hyp_meta_entry['score']

                new_hyp = prev_hyp.clone_and_apply_action_info(action_info)
                new_hyp.score = new_hyp_score
                new_hyp.inputs.extend(prev_hyp.inputs)


                if new_hyp.is_valid is False:
                    continue


                if new_hyp.completed:
                    completed_beams.append(new_hyp)
                else:
                    new_beams.append(new_hyp)
                    live_hyp_ids.append(prev_hyp_id)

            if live_hyp_ids:
                h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
                att_tm1 = att[live_hyp_ids]
                beams = new_beams
                t += 1
            else:
                break

        if len(completed_beams) == 0:
            print('Predict lf failed!!!\n')
            return []

        completed_beams.sort(key=lambda hyp: -hyp.score)

        return completed_beams



    def train_action_one_turn(self,utterance,utterance_states,final_utterance_state,
                              interaction,schema_states,turn_states,utterance_states_list_store,
                              previous_lf_queries,previous_lf_query_states):

        last_cell = final_utterance_state[0][-1].unsqueeze(0)
        dec_init_vec = self.init_decoder_state(last_cell)
        h_tm1 = dec_init_vec
        tgt_action_probs = []
        # action_states = []

        gold_token_count = 0

        turn_enable = np.zeros(shape=utterance_states.size(0), dtype=np.int32)
        valueS_enable = np.zeros(shape=utterance_states.size(0), dtype=np.int32)

        for t in range(utterance.action_num):
            if t == 0:
                x = self.new_tensor(1, self.lf_decoder_lstm.input_size).zero_()
            else:
                if t < utterance.action_num:
                    action_tm1 = utterance.target_actions[t - 1]
                    if type(action_tm1) in [define_rule.Root1,
                                            define_rule.Root,
                                            define_rule.Sel,
                                            define_rule.Filter,
                                            define_rule.Sup,
                                            define_rule.N,
                                            define_rule.Order,
                                            ]:
                        a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                    else:
                        if isinstance(action_tm1, define_rule.TC):
                            a_tm1_embed = self.schema_rnn_input(interaction.column_name_embeddings[action_tm1.id_c])
                        elif isinstance(action_tm1, define_rule.A):
                            a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                        elif isinstance(action_tm1, define_rule.Turn):
                            a_tm1_embed = self.turn_rnn_input(turn_states[0][ action_tm1.id_c ])
                        elif isinstance(action_tm1, define_rule.ValueS):
                            a_tm1_embed = self.valueS_rnn_input(utterance_states_list_store[turn_enable[0]][action_tm1.id_c])
                        elif isinstance(action_tm1, define_rule.Distance):
                            a_tm1_embed = self.distance_rnn_input(utterance_states_list_store[turn_enable[0]][valueS_enable[0]+action_tm1.id_c])
                        else:
                            # print(action_tm1, 'not implement')
                            # quit()
                            # a_tm1_embed = zero_action_embed
                            pass

                inputs = [a_tm1_embed]
                if t < utterance.action_num:
                    action_tm1 = utterance.target_actions[t - 1]
                    pre_type = self.type_embed.weight[self.grammar.type2id[type(action_tm1)]]

                inputs.append(pre_type)
                inputs.append(att_tm1)

                x = torch.cat(inputs, dim=-1).view(1, -1)

            (h_t, cell_t),state_and_attn,att = self.decoder_lf_one_step_query(x, h_tm1, utterance_states,schema_states,
                                                             self.lf_decoder_lstm,previous_lf_query_states)

            # predict rule
            att_t = torch.tanh(self.lf_att_vec_linear(state_and_attn))
            att_t = self.dropout_layer(att_t)
            apply_rule_prob = torch.softmax(self.production_readout(att_t), dim=-1).squeeze(0)

            if att_t.dim()==1:
                att_t = att_t.unsqueeze(0)

            schema_appear_mask = interaction.schema_appear_mask

            schema_appear_mask = torch.from_numpy(schema_appear_mask) #[B, len]
            if schema_states.is_cuda:
                schema_appear_mask = schema_appear_mask.cuda()

            if self.args.use_column_pointer:
                gate = torch.sigmoid(self.prob_att(att_t)) #[B,1]
                #[B,len_schema]
                schema_weight = self.schema_pointer_net(schema_states,att_t.unsqueeze(0)) * schema_appear_mask * gate + \
                    self.schema_pointer_net(schema_states,att_t.unsqueeze(0)) * (1 - schema_appear_mask) * (1 - gate)
            else:
                schema_weight = self.schema_pointer_net(schema_states,att_t.unsqueeze(0))

            schema_weight = torch.softmax(schema_weight, dim=-1).squeeze(0)


            turn_weight  = self.turn_pointer_net(turn_states,att_t.unsqueeze(0))
            turn_weight = torch.softmax(turn_weight, dim=-1).squeeze(0)

            noise = 1e-8

            if att_t.dim() == 1:
                att_t = att_t.unsqueeze(0)

            # # TODO use previous sketck query
            # intermediate_state = self._get_intermediate_state_lf(state_and_attn, dropout_amount=self.dropout_ratio)
            if self.args.use_previous_query and len(previous_lf_queries) > 0:
                if self.args.use_copy_switch:
                    copy_switch = self._get_copy_switch(att_t)

                assert len(previous_lf_queries) == 1, 'debug mode'
                previous_lf_query_state = previous_lf_query_states[-1]

                previous_lf_query_state = torch.stack(previous_lf_query_state, dim=0).repeat(1, 1, 1)
                # [B, len_query]
                query_scores = self.query_pointer_net(previous_lf_query_state, att_t.unsqueeze(0))
                query_scores = torch.softmax(query_scores, dim=-1).squeeze(0)

                query_scores = query_scores * copy_switch
                apply_rule_prob = apply_rule_prob * (1 - copy_switch)
                schema_weight = schema_weight * (1-copy_switch)
                turn_weight = turn_weight * (1 - copy_switch)


            action_t = utterance.target_actions[t]

            if len(previous_lf_queries) > 0:

                previous_lf_query = previous_lf_queries[-1]
                query_scores_list = []
                for i,item in enumerate(previous_lf_query):
                    if isinstance(action_t,define_rule.A) or \
                            isinstance(action_t,define_rule.TC) or \
                            isinstance(action_t, define_rule.Turn) or \
                            isinstance(action_t, define_rule.ValueS) or \
                            isinstance(action_t, define_rule.Distance):
                        if str(item) == str(action_t) and item.parent == action_t.parent:
                            query_scores_list.append(query_scores[i] + noise)
                if query_scores_list != []:
                    tgt_action_probs.extend(query_scores_list)

            noise = 1e-11
            action_probs_map = lambda x: torch.clamp(x, noise)

            if isinstance(action_t, define_rule.TC):
                interaction.update_schema_appear_mask(action_t.id_c)
                act_prob_t_i = schema_weight[action_t.id_c]
                tgt_action_probs.append( action_probs_map(act_prob_t_i) )
                gold_token_count +=1
            elif isinstance(action_t, define_rule.A):
                act_prob_t_i = apply_rule_prob[self.grammar.prod2id[action_t.production]]
                tgt_action_probs.append( action_probs_map(act_prob_t_i) )
                gold_token_count +=1
            elif isinstance(action_t, define_rule.Turn):

                turn_enable[0] = action_t.id_c

                act_prob_t_i = turn_weight[action_t.id_c]
                tgt_action_probs.append( action_probs_map(act_prob_t_i) )
                gold_token_count += 1

            elif isinstance(action_t, define_rule.ValueS):

                valueS_enable[0] = action_t.id_c
                utter_state = utterance_states_list_store[turn_enable[0]]

                assert action_t.id_c < len(utter_state)
                single_states = torch.stack(utter_state,dim=0).unsqueeze(0)
                valueS_weight = self.valueS_pointer_net(single_states, att_t.unsqueeze(0))

                valueS_weight = torch.softmax(valueS_weight, dim=-1).squeeze(0)

                assert valueS_weight.size(0) == single_states.size(1)

                if self.args.use_previous_query and len(previous_lf_queries) > 0:
                    if self.args.use_copy_switch:
                        valueS_weight = valueS_weight * (1 - copy_switch)

                act_prob_t_i = valueS_weight[action_t.id_c]
                tgt_action_probs.append(action_probs_map(act_prob_t_i))
                gold_token_count += 1


            elif isinstance(action_t, define_rule.Distance):

                utter_state = utterance_states_list_store[turn_enable[0]]

                start_loc =  valueS_enable[0]
                end_loc = min(start_loc + action_t.max_len, len(utter_state))
                utter_state = utter_state[start_loc : end_loc]
                dist_states = torch.stack(utter_state, dim=0).unsqueeze(0)
                dist_weight = self.distance_pointer_net(dist_states, att_t.unsqueeze(0))
                dist_weight = torch.softmax(dist_weight, dim=-1).squeeze(0)

                assert action_t.id_c < dist_weight.size(0)

                if self.args.use_previous_query and len(previous_lf_queries) > 0:
                    if self.args.use_copy_switch:
                        dist_weight = dist_weight * (1 - copy_switch)

                dist_prob_t_i = dist_weight[action_t.id_c]
                tgt_action_probs.append(action_probs_map(dist_prob_t_i))
                gold_token_count += 1

            else:
                pass

            h_tm1 = (h_t, cell_t)
            att_tm1 = att.squeeze()

        lf_prob_var = -torch.stack(tgt_action_probs, dim=0).log().sum()

        return lf_prob_var,gold_token_count


    def train_sketch_one_turn(self,utterance,utterance_states,final_utterance_state,previous_sketch_queries,previous_sketch_query_states):
        # TODO (1,7,350) -> (1, 7,300)

        last_cell = final_utterance_state[0][-1].unsqueeze(0)

        # TODO 会话编码最后的状态，作为解码器初始输入，（c,h)  h初始为0
        dec_init_vec = self.init_decoder_state(last_cell)
        h_tm1 = dec_init_vec
        action_probs = []

        gold_token_count = 0

        for t in range(utterance.sketch_num):
            if t == 0:
                x = self.new_tensor(1, self.sketch_decoder_lstm.input_size).zero_()
            else:
                if t < utterance.sketch_num:
                    action_tm1 = utterance.sketch_actions[t - 1]
                    if type(action_tm1) in [define_rule.Root1,
                                            define_rule.Root,
                                            define_rule.Sel,
                                            define_rule.Filter,
                                            define_rule.Sup,
                                            define_rule.N,
                                            define_rule.Order]:
                        # TODO torch.Size([128]) 每个action的embedding
                        a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                    else:
                        raise ValueError('only for sketch')
                else:
                    raise ValueError('always t < sketch num,due to batch size = 1')

                inputs = [a_tm1_embed]

                if t < utterance.sketch_num:
                    action_type = utterance.sketch_actions[t - 1]
                    pre_type = self.type_embed.weight[self.grammar.type2id[type(action_type)]]

                inputs.append(pre_type)
                inputs.append(att_tm1)

                x = torch.cat(inputs, dim=-1).view(1, -1)

            (h_t, cell_t),state_and_attn,att = self.decoder_sketch_one_step_query(x, h_tm1, utterance_states,
                                                             self.sketch_decoder_lstm,previous_sketch_query_states)

            # predict rule
            att_t = torch.tanh(self.sketch_att_vec_linear(state_and_attn))
            att_t = self.dropout_layer(att_t)
            # TODO torch.Size([46])
            apply_rule_prob = torch.softmax(self.production_readout(att_t), dim=-1).squeeze(0)
            noise = 1e-8

            if att_t.dim() == 1:
                att_t = att_t.unsqueeze(0)

            # TODO use previous sketck query
            # intermediate_state = self._get_intermediate_state_sketch(state_and_attn, dropout_amount=self.dropout_ratio)
            if self.args.use_previous_query and len(previous_sketch_queries) > 0:
                if self.args.use_copy_switch:
                    copy_switch = self._get_copy_switch(att_t)

                assert len(previous_sketch_queries) == 1,'debug mode'
                previous_sketch_query_state = previous_sketch_query_states[-1]

                previous_sketch_query_state = torch.stack(previous_sketch_query_state, dim=0).repeat(1, 1, 1)

                # [B, len_query]
                query_scores = self.query_pointer_net(previous_sketch_query_state, att_t.unsqueeze(0))
                query_scores = torch.softmax(query_scores, dim=-1).squeeze(0)


                query_scores = query_scores * copy_switch
                apply_rule_prob = apply_rule_prob * (1 - copy_switch)


            action_t = utterance.sketch_actions[t]

            if len(previous_sketch_queries) > 0:

                previous_sketch_query = previous_sketch_queries[-1]
                query_scores_list = []
                for i,item in enumerate(previous_sketch_query):
                    if str(item) == str(action_t) and item.parent == action_t.parent:
                        query_scores_list.append(query_scores[i] + noise)

                action_probs.extend(query_scores_list)

            act_prob_t_i = apply_rule_prob[self.grammar.prod2id[action_t.production]]

            action_probs.append(act_prob_t_i + noise)

            gold_token_count += 1

            h_tm1 = (h_t, cell_t)
            att_tm1 = att


        sketch_prob_var = -torch.stack(action_probs, dim=0).log().sum()

        return sketch_prob_var,gold_token_count


    # def _get_intermediate_state_lf(self, state, dropout_amount=0.):
    #     intermediate_state = torch.tanh(torch_utils.linear_layer(state, self.state_transform_weights_lf))
    #     return F.dropout(intermediate_state, dropout_amount)
    #
    # def _get_intermediate_state_sketch(self, state, dropout_amount=0.):
    #     intermediate_state = torch.tanh(torch_utils.linear_layer(state, self.state_transform_weights_sketch))
    #     return F.dropout(intermediate_state, dropout_amount)


    # def _get_query_token_scorer(self, state):
    #     scorer = torch.t(torch_utils.linear_layer(state, self.query_token_weights))
    #     return scorer


    # def score_query_tokens(self,previous_query, previous_query_states, scorer):
    #     scores = torch.t(torch.mm(torch.t(scorer), previous_query_states))  # num_tokens x 1
    #     if scores.size()[0] != len(previous_query):
    #         raise ValueError(
    #             "Got " + str(scores.size()[0]) + " scores for " + str(len(previous_query)) + " query tokens")
    #     return scores, previous_query

    def _get_copy_switch(self, state):
        copy_switch = torch.sigmoid(torch_utils.linear_layer(state, self.state2copyswitch_transform_weights))

        return copy_switch.squeeze()

    # def _get_copy_switch(self, state):
    #     copy_switch = torch.sigmoid(torch_utils.linear_layer(state, self.state2copyswitch_transform_weights_lf))

        # return copy_switch.squeeze()

    def get_utterance_attention(self, final_utterance_states_c, final_utterance_states_h, final_utterance_state, num_utterances_to_keep):
        # self-attention between utterance_states
        final_utterance_states_c.append(final_utterance_state[0][0])
        final_utterance_states_h.append(final_utterance_state[1][0])
        #TODO 1,2,3 torch.Size([300])
        # print(len(final_utterance_states_c))
        # print(final_utterance_states_h[0].size())
        final_utterance_states_c = final_utterance_states_c[-num_utterances_to_keep:]
        final_utterance_states_h = final_utterance_states_h[-num_utterances_to_keep:]

        #TODO 当前会话与所有会话的attention
        attention_result = self.utterance_attention_module(final_utterance_states_c[-1], final_utterance_states_c)
        final_utterance_state_attention_c = final_utterance_states_c[-1] + attention_result.vector.squeeze()

        attention_result = self.utterance_attention_module(final_utterance_states_h[-1], final_utterance_states_h)
        final_utterance_state_attention_h = final_utterance_states_h[-1] + attention_result.vector.squeeze()

        final_utterance_state = ([final_utterance_state_attention_c],[final_utterance_state_attention_h])

        return final_utterance_states_c, final_utterance_states_h, final_utterance_state


    def decoder_sketch_one_step_query(self,x,h_tm1,utterance_states, decoder,previous_sketch_query_states):
        '''
            x : 1） 来自上一轮 的 action (ROOT(0),ROOT1(0),...),  2）即输出的att_t ，    3）action的类型（ROOT || SEL,...），
            h_tm1 : 来自question的编码状态

            att_t ： 由  h_t,_ = LSTMCell(x, h_tm1) 得到编码状态，再由h_t与 question token 做attention，得到句子级向量，
                    与 h_t拼接成（B,600) ,做linear->(B,300) -> tanh ->dropout ,得到att_t
        '''
        #TODO torch.Size([1, 300]) torch.Size([1, 300])

        h_t, cell_t = decoder(x, h_tm1)

        utterance_states_list = [x.squeeze() for x in
                                 list(torch.split(utterance_states.squeeze(), split_size_or_sections=1, dim=0))]

        # TODO torch.Size([350])
        utterance_attention_results = self.attention_module_utter(h_t.squeeze(), utterance_states_list)

        if self.args.use_query_attention:
            if len(previous_sketch_query_states) > 0:
                # TODO (1,300) -> (300)
                query_attention_results = self.query_attention_module(h_t.squeeze(), previous_sketch_query_states[-1])
            else:
                query_attention_results = self.start_query_attention_vector
                query_attention_results = AttentionResult(None, None, query_attention_results)

        if self.args.use_query_attention:
            state_and_attn = torch.cat([h_t.squeeze(),utterance_attention_results.vector, query_attention_results.vector], dim=0)
            ret_att = torch.cat([utterance_attention_results.vector, query_attention_results.vector], dim=0)
        else:
            state_and_attn = torch.cat([h_t.squeeze(),utterance_attention_results.vector],dim=0)

        return (h_t, cell_t),state_and_attn,ret_att


    def decoder_lf_one_step_query(self,x,h_tm1,utterance_states, schema_states,decoder,previous_sketch_query_states):

        h_t, cell_t = decoder(x, h_tm1)

        utterance_states_list2 = [x.squeeze() for x in
                                 list(torch.split(utterance_states.squeeze(), split_size_or_sections=1, dim=0))]

        # TODO torch.Size([350])
        utterance_attention_results = self.attention_module_utter(h_t.squeeze(), utterance_states_list2)

        schema_states_list = [x.squeeze() for x in
                                 list(torch.split(schema_states.squeeze(), split_size_or_sections=1, dim=0))]

        # TODO torch.Size([350])
        schema_attention_results = self.attention_module_schema(h_t.squeeze(), schema_states_list)

        if self.args.use_query_attention:
            if len(previous_sketch_query_states) > 0:
                # TODO (1,300) -> (300)
                query_attention_results = self.query_attention_module(h_t.squeeze(), previous_sketch_query_states[-1])
            else:
                query_attention_results = self.start_query_attention_vector
                query_attention_results = AttentionResult(None, None, query_attention_results)

        if self.args.use_query_attention:

            state_and_attn = torch.cat([h_t.squeeze(),utterance_attention_results.vector, schema_attention_results.vector,query_attention_results.vector], dim=0)
            ret_att = torch.cat([utterance_attention_results.vector, schema_attention_results.vector,query_attention_results.vector], dim=0)
        else:
            state_and_attn = torch.cat([h_t.squeeze(),utterance_attention_results.vector,schema_attention_results.vector],dim=0)
        # if state_and_attn.dim() == 1:
        #     state_and_attn = state_and_attn.unsqueeze(0)


        return (h_t, cell_t),state_and_attn,ret_att


    def init_decoder_state(self, enc_last_cell):
        h_0 = self.decoder_cell_init(enc_last_cell)
        h_0 = torch.tanh(h_0)

        return h_0, self.new_tensor(h_0.size()).zero_()

    def encode_schema_self_attention(self, schema_states):
        #TODO torch.stack(schema_states,dim=0) == (N,300)
        # print(torch.stack(schema_states,dim=0).size()) == (23,300)
        schema_self_attention = self.schema2schema_attention_module(torch.stack(schema_states,dim=0), schema_states).vector
        #TODO torch.Size([300, 23])
        # print(schema_self_attention.size())
        if schema_self_attention.dim() == 1:
            schema_self_attention = schema_self_attention.unsqueeze(1)
        residual_schema_states = list(torch.split(schema_self_attention, split_size_or_sections=1, dim=1))
        # torch.Size([300, 1])
        # print(residual_schema_states[0].size())
        residual_schema_states = [schema_state.squeeze() for schema_state in residual_schema_states]

        new_schema_states = [schema_state+residual_schema_state for schema_state, residual_schema_state in zip(schema_states, residual_schema_states)]

        return new_schema_states

    def get_bert_encoding(self, input_sequence, interaction, discourse_state, dropout):

        num_sequence = len(input_sequence)
        flat_idxes = []
        if isinstance(input_sequence[0],list):
            flat_input_sequence = []
            idx = 0
            for x in input_sequence:
                flat_idxes.append(list(range(idx,idx+len(x))))
                flat_input_sequence.extend(x)
                idx += len(x)
            input_sequence = flat_input_sequence


        # TODO utterance_states :list ,每个元素是 question 的embedding ，size = 768
        # TODO schema_token_states :list,每个元素是 table——column，也是个list（有多个名字，eg：table . column）,再每个元素才是embedding
        utterance_states, schema_token_states = utils_bert.get_bert_encoding(self.bert_config, self.model_bert, self.tokenizer, input_sequence, interaction.column_names_embedder_input, bert_input_version='v1', num_out_layers_n=1, num_out_layers_h=1)
        if flat_idxes:
            new_utterance_states = []
            for idxes in flat_idxes:
                tmp_states = []
                for idx in idxes:
                    tmp_states.append(utterance_states[idx])
                one_states = torch.mean(torch.stack(tmp_states,dim=0),dim=0)
                new_utterance_states.append(one_states)
            utterance_states = new_utterance_states

        assert len(utterance_states) == num_sequence
        assert len(interaction.column_names_embedder_input) == len(schema_token_states)
        # turn_idx = torch.tensor([0])
        # valueS_idx = torch.tensor([0])

        # if self.args.cuda:
        #     turn_idx = turn_idx.cuda()
        #     valueS_idx = valueS_idx.cuda()
        # print(turn_idx.is_cuda)
        if self.args.discourse_level_lstm:
            utterance_token_embedder = lambda x, turn_idx, valueS_idx: torch.cat([x, discourse_state], dim=0)
        else:
            utterance_token_embedder = lambda x, turn_idx, valueS_idx: x

        #TODO utterance state + discourse state -> BiLSTM
        if dropout:
            '''
                final_utterance_state : (cell_memories, hidden_states)
                utterance_states : list , len() = question token,每个元素300
            '''
            final_utterance_state, utterance_states = self.utterance_encoder(
                utterance_states,
                utterance_token_embedder,
                dropout_amount=self.dropout_ratio)
            # TODO torch.Size([7, 300])
            # print(torch.stack(utterance_states,dim=0).size())
        else:
            final_utterance_state, utterance_states = self.utterance_encoder(
                utterance_states,
                utterance_token_embedder)

        #TODO schema state , 取每列的最后一个状态（最简单直接取平均）
        schema_states = []
        for schema_token_states1 in schema_token_states:
            if dropout:
                #TODO schema_states_one is [table . col] output,len(schema_states_one) = 1,4,3,3,6..
                # we just use the final hidden state `final_schema_state_one`
                final_schema_state_one, schema_states_one = self.schema_encoder(schema_token_states1, lambda x,turn_idx, valueS_idx: x,
                                                                                dropout_amount=self.dropout_ratio)
            else:
                final_schema_state_one, schema_states_one = self.schema_encoder(schema_token_states1, lambda x,turn_idx, valueS_idx: x)

            # final_schema_state_one: 1 means hidden_states instead of cell_memories, -1 means last layer
            schema_states.append(final_schema_state_one[1][-1])  # TODO 取出hidden_size 最后的输出
        # TODO torch.Size([46, 300])
        # print(torch.stack(schema_states,dim=0).size())

        interaction.set_column_name_embeddings(schema_states)

        # self-attention over schema_states
        if self.args.use_schema_self_attention:
            # TODO torch.Size([46, 300])
            schema_states = self.encode_schema_self_attention(schema_states)

        return final_utterance_state, utterance_states, schema_states
