# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# -*- coding: utf-8 -*-


import json
import time
import subprocess
import copy
import numpy as np
import os
import torch
import shutil

from nltk.stem import WordNetLemmatizer

#from src.dataset import Example
from src.rule import lf
from src.rule.semQL import Sup, Sel, Order, Root, Filter, A, N, C, T, Root1
from src.rule import semQL as define_rule
from src.rule.sem_utils import load_schemas
from src.sem2SQL import transform,parse_sql_to_std
import src.postprocess_eval as post_eval
from src.models.bert import utils_bert

wordnet_lemmatizer = WordNetLemmatizer()


def load_word_emb(file_name, use_small=False):
    print ('Loading word embedding from %s'%file_name)
    ret = {}
    with open(file_name) as inf:
        for idx, line in enumerate(inf):
            if (use_small and idx >= 1000):
                break
            info = line.strip().split(' ')
            if info[0].lower() not in ret:
                ret[info[0]] = np.array(list(map(lambda x:float(x), info[1:])))
    return ret

def lower_keys(x):
    if isinstance(x, list):
        return [lower_keys(v) for v in x]
    elif isinstance(x, dict):
        return dict((k.lower(), lower_keys(v)) for k, v in x.items())
    else:
        return x

def get_table_colNames(tab_ids, tab_cols):
    table_col_dict = {}

    for ci, cv in zip(tab_ids, tab_cols):
        if ci != -1:
            table_col_dict[ci] = table_col_dict.get(ci, []) + cv

    #TODO 注意， 列都被展开成 一维list了
    # {0: ['department', 'id', 'name', 'creation', 'ranking', 'budget', 'in', 'billion', 'num', 'employee'],
    # 1: ['head', 'id', 'name', 'born', 'state', 'age'],
    # 2: ['department', 'id', 'head', 'id', 'temporary', 'acting']}
    # print(table_col_dict)
    result = []
    for ci in range(len(table_col_dict)):
        result.append(table_col_dict[ci])
    return result

def get_col_table_dict(tab_cols, tab_ids, sql):
    table_dict = {}

    for c_id, c_v in enumerate(sql['col_set']):
        for cor_id, cor_val in enumerate(tab_cols):
            if c_v == cor_val:
                table_dict[tab_ids[cor_id]] = table_dict.get(tab_ids[cor_id], []) + [c_id]
    # TODO table_dict : {表名:[列名 list], }{0: [1, 2, 3, 4, 5, 6], 1: [2, 7, 8, 9], 2: [1, 7, 10], -1: [0]}

    col_table_dict = {}
    for key_item, value_item in table_dict.items():
        for value in value_item:
            col_table_dict[value] = col_table_dict.get(value, []) + [key_item]

    col_table_dict[0] = [x for x in range(len(table_dict) - 1)]
    #TODO col_table_dict : {列名： [表名 list], ... }
    # {0: [0, 1, 2], 1: [0, 2], 2: [0, 1], 3: [0], 4: [0], 5: [0], 6: [0], 7: [1, 2], 8: [1], 9: [1], 10: [2]}
    return col_table_dict

def epoch_train_with_interaction(epoch,log,model,optimizer,bert_optimizer,interaction_batches,args):
    model.train()

    loss_sum = 0.

    average_sketch_loss = []
    average_lf_loss = []
    average_total_loss = []

    for i,interaction_batch in enumerate(interaction_batches):
        assert len(interaction_batch) == 1

        interaction = interaction_batch[0]
        if interaction.interaction_list == []:
            continue
        #TODO Attribute dict_keys(['interaction', 'tab_cols', 'col_iter', 'tab_ids', 'index', 'table_names'])
        # print(interaction.__dict__.keys())

        batch_loss,sk,lf = model(i,epoch,interaction,optimizer,bert_optimizer)
       
        average_sketch_loss.append(sk)
        average_lf_loss.append(lf)
        average_total_loss.append(batch_loss)

        if (i+1) % 1000 ==0:
            sk = torch.mean(torch.tensor(average_sketch_loss,dtype=torch.float))
            lf = torch.mean(torch.tensor(average_lf_loss,dtype=torch.float))
            total = torch.mean(torch.tensor(average_total_loss,dtype=torch.float))
            log_str = '#interaction id:#{}\t,total_loss:{:.3f},\tsketch_loss:{:.3f},\tlf_loss:{:.3f}'.format(i+1, total,sk,lf)
            print(log_str)
            log.put(log_str)

            average_total_loss = []
            average_lf_loss= []
            average_sketch_loss = []


        loss_sum += batch_loss
    #print('All question:',utils_bert.total_count)
    #print('Each part:',utils_bert.total_list)

    total_loss = loss_sum / len(interaction_batches)

    return total_loss



def epoch_acc_with_interaction(epoch,model, valid_batchs,args,beam_size=1,use_predicted_queries=False,log=None):
    model.eval()
    sketch_correct = 0
    lf_correct  = 0
    interaction_lf_correct_total = 0

    num_interactions = 0
    num_utterances = 0
    if log:
        log.put('='*20+str(epoch)+'='*20+'\n')
    for i, interaction in enumerate(valid_batchs):
        if log:
            log.put('#The {}th interaction#'.format(i))
        with torch.no_grad():
            if use_predicted_queries:
                pred_lf_actions,pred_sketch_actions = model.predict_with_predicted_queries(interaction,beam_size=beam_size)
            else:
                pred_lf_actions,pred_sketch_actions = model.predict_with_gold_queries(interaction,beam_size=beam_size)

            num_interactions += 1
            num_utterances += len(pred_lf_actions)
            interaction_lf_correct = 0
            for k,(pred_lf_action, pred_sketch_action,utterance) in enumerate(zip(pred_lf_actions,pred_sketch_actions, interaction.interaction_list)):

                truth_sketch_action = " ".join([str(x) for x in utterance.sketch_actions])
                truth_lf_action = " ".join([str(x) for x in utterance.target_actions])
                pred_sketch_bool = 'No'
                pred_lf_bool = 'No'
                if pred_lf_action != []:
                    pred_lf = " ".join([str(x) for x in pred_lf_action])
                    pred_sketch = " ".join([str(x) for x in pred_sketch_action])
                    if truth_lf_action == pred_lf:
                        lf_correct += 1
                        interaction_lf_correct += 1
                        pred_lf_bool = 'Yes'
                    if truth_sketch_action == pred_sketch:
                        sketch_correct +=1
                        pred_sketch_bool = 'Yes'
                else:
                    pred_lf = 'None'
                    pred_sketch = 'None'
                if log:
                    log_str = '[the {}th utterance]\nsketch pred : {}\nsketch grth : {}\t{}\n' \
                              'lf pred : {}\nlf grth : {}\t{}\n'.format(k,pred_sketch,truth_sketch_action,pred_sketch_bool,pred_lf,truth_lf_action,pred_lf_bool)
                    log.put(log_str)
            if interaction_lf_correct == len(pred_lf_actions):
                interaction_lf_correct_total += 1



    sketch_acc = sketch_correct / num_utterances
    lf_acc = lf_correct / num_utterances

    interaction_lf_acc = interaction_lf_correct_total / num_interactions

    return sketch_acc,lf_acc , interaction_lf_acc


def epoch_acc_with_interaction_save_json(epoch,model, valid_batchs,args,beam_size=1,use_predicted_queries=False):
    model.eval()
    sketch_correct = 0
    lf_correct  = 0
    interaction_lf_correct_total = 0

    num_interactions = 0
    num_utterances = 0
    jsonf_data = []
    for i, interaction in enumerate(valid_batchs):
        full_interaction = interaction.full_interaction

        with torch.no_grad():
            if use_predicted_queries:
                pred_lf_actions,pred_sketch_actions = model.predict_with_predicted_queries(interaction,beam_size=beam_size)
            else:
                pred_lf_actions,pred_sketch_actions = model.predict_with_gold_queries(interaction,beam_size=beam_size)

            num_interactions += 1
            num_utterances += len(pred_lf_actions)
            interaction_lf_correct = 0
            for k, (pred_lf_action, pred_sketch_action, utterance, utter_dict) in enumerate(
                    zip(pred_lf_actions, pred_sketch_actions, interaction.interaction_list,
                        full_interaction['interaction'])):

                truth_sketch_action = " ".join([str(x) for x in utterance.sketch_actions])
                truth_lf_action = " ".join([str(x) for x in utterance.target_actions])

                target_CT_actions = convert_TC_to_C_and_T(utterance.target_actions,
                                                          full_interaction['columns_names_embedder'],
                                                          full_interaction['columns_names_embedder_idxes'])
                utter_dict['target_lf'] = " ".join([str(x) for x in target_CT_actions])
                utter_dict['pred_sketch'] = ""
                utter_dict['pred_lf'] = ""
                if pred_lf_action != []:

                    predict_lf_CT_actions = convert_TC_to_C_and_T(pred_lf_action,
                                                                  full_interaction['columns_names_embedder'],
                                                                  full_interaction['columns_names_embedder_idxes'])
                    pred_lf = " ".join([str(x) for x in pred_lf_action])
                    pred_sketch = " ".join([str(x) for x in pred_sketch_action])
                    utter_dict['pred_sketch'] = pred_sketch
                    utter_dict['pred_lf'] = " ".join([str(x) for x in predict_lf_CT_actions])
                    if truth_lf_action == pred_lf:
                        lf_correct += 1
                        interaction_lf_correct += 1
                    if truth_sketch_action == pred_sketch:
                        sketch_correct += 1
            if interaction_lf_correct == len(pred_lf_actions):
                interaction_lf_correct_total += 1

        jsonf_data.append(full_interaction)

    sketch_acc = sketch_correct / num_utterances
    lf_acc = lf_correct / num_utterances

    interaction_lf_acc = interaction_lf_correct_total / num_interactions

    return jsonf_data, sketch_acc, lf_acc, interaction_lf_acc

def get_schema_tokens(table_schema):
    column_names_surface_form = []
    column_names = []
    column_names_original = table_schema['column_names_original']
    table_names = table_schema['table_names']
    table_names_original = table_schema['table_names_original']
    for i, (table_id, column_name) in enumerate(column_names_original):
      if table_id >= 0:
        table_name = table_names_original[table_id]
        column_name_surface_form = '{}.{}'.format(table_name,column_name)
      else:
        # this is just *
        column_name_surface_form = column_name
      column_names_surface_form.append(column_name_surface_form.lower())
      column_names.append(column_name.lower())

    # also add table_name.*
    for table_name in table_names_original:
      column_names_surface_form.append('{}.*'.format(table_name.lower()))

    return column_names_surface_form, column_names


def semQL2SQL_question_and_interaction_match(jsonf,args):
    schemas = load_schemas(args)
    datas = jsonf

    predict_f = open("prediction.json", "w")

    for i, interaction in enumerate(datas):
        utter_list = []
        for utter_idx, utter in enumerate(interaction['interaction']):

            utter_list.append(utter['origin_utterance_arg'])
            utter_dict = {}
            utter_dict['identifier'] = datas[i]['database_id'] + '/' + str(i)
            utter_dict['database_id'] = datas[i]['database_id']
            utter_dict['interaction_id'] = str(i)
            utter_dict['index_in_interaction'] = utter_idx
            utter_dict['input_seq'] = utter['origin_utterance_toks']

            schema_tokens, column_names = get_schema_tokens(schemas[datas[i]['database_id']])

            result = transform(datas[i], utter, utter_list, schemas[datas[i]['database_id']])

            if result[0]!='':
                t = result[0].replace('.', ' . ').replace(',', ' , ').replace('(', ' ( ').replace(')', ' ) ')
                m = [x for x in t.split(' ') if x != '']

                try:
                    pred_turn_sql_parse = parse_sql_to_std(m, column_names, schema_tokens,
                                                       schemas[datas[i]['database_id']])
                    utter_dict['flat_prediction'] = pred_turn_sql_parse.split(' ')
                except Exception as e:
                    utter_dict['flat_prediction'] = ['']
                # if 't1' in utter_dict['flat_prediction'] or 't2' in utter_dict['flat_prediction'] or 't3' in utter_dict['flat_prediction'] or 't4' in utter_dict['flat_prediction'] or 't5' in utter_dict['flat_prediction']:
                #     utter_dict['flat_prediction'] = ['']
            else:
                utter_dict['flat_prediction'] = ['']
            # print(utter.keys())
            # target_turn_sql_parse = parse_sql_to_std(utter['query_toks_no_value'], column_names, schema_tokens,
            #                                          schemas[datas[i]['database_id']])

            # utter_dict['flat_gold_queries'] = [target_turn_sql_parse.split(' ')]

            predict_f.write(json.dumps(utter_dict) + '\n')

    predict_f.close()

    pred_file = "prediction.json"
    table_schema_path = os.path.join(args.dataset,'tables.json')
    gold_path = os.path.join('sparc','dev_gold.txt')
    db_path =  os.path.join('sparc','database')
    remove_from = True
    dataset = 'sparc'
    database_schema = post_eval.read_schema(table_schema_path)

    predictions = post_eval.read_prediction(pred_file)

    postprocess_sqls = post_eval.postprocess(predictions, database_schema, remove_from)

    question_match, interaction_match = post_eval.write_and_evaluate(postprocess_sqls, db_path, table_schema_path, gold_path, dataset)
    # os.system(command)
    #eval_output = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
    return question_match,interaction_match


def convert_TC_to_C_and_T(lf_action,embedder_input,embedder_input_idxes):

    convert_action_list = []
    for action in lf_action:
        if type(action) != define_rule.TC:
            convert_action_list.append(action)
        else:
            t_id,c_id = embedder_input_idxes[action.id_c]
            convert_action_list.append(C(c_id))
            convert_action_list.append(T(t_id))
    return convert_action_list

def eval_with_interaction_save_json(model,valid_batchs,beam_size=1):
    model.eval()
    sketch_correct = 0
    lf_correct = 0
    interaction_lf_correct_total = 0

    num_interactions = 0
    num_utterances = 0
    jsonf_data = []
    for i, interaction in enumerate(valid_batchs):
        full_interaction = interaction.full_interaction
        with torch.no_grad():

            pred_lf_actions, pred_sketch_actions = model.predict_with_predicted_queries(interaction,beam_size=beam_size)

            num_interactions += 1
            num_utterances += len(pred_lf_actions)
            interaction_lf_correct = 0
            for k, (pred_lf_action, pred_sketch_action, utterance,utter_dict) in enumerate(
                    zip(pred_lf_actions, pred_sketch_actions, interaction.interaction_list,full_interaction['interaction'])):

                truth_sketch_action = " ".join([str(x) for x in utterance.sketch_actions])
                truth_lf_action = " ".join([str(x) for x in utterance.target_actions])
            
                target_CT_actions = convert_TC_to_C_and_T(utterance.target_actions,full_interaction['columns_names_embedder'],full_interaction['columns_names_embedder_idxes'])
                utter_dict['target_lf'] = " ".join([str(x) for x in target_CT_actions])

                utter_dict['pred_sketch'] = ""
                utter_dict['pred_lf'] = ""
                if pred_lf_action != []:
                    predict_lf_CT_actions = convert_TC_to_C_and_T(pred_lf_action,
                                                              full_interaction['columns_names_embedder'],
                                                              full_interaction['columns_names_embedder_idxes'])

                    pred_lf = " ".join([str(x) for x in pred_lf_action])
                    pred_sketch = " ".join([str(x) for x in pred_sketch_action])
                    utter_dict['pred_sketch'] = pred_sketch
                    utter_dict['pred_lf'] = " ".join([str(x) for x in predict_lf_CT_actions])
                    if truth_lf_action == pred_lf:
                        lf_correct += 1
                        interaction_lf_correct += 1
                    if truth_sketch_action == pred_sketch:
                        sketch_correct += 1
            if interaction_lf_correct == len(pred_lf_actions):
                interaction_lf_correct_total += 1
        jsonf_data.append(full_interaction)

    sketch_acc = sketch_correct / num_utterances
    lf_acc = lf_correct / num_utterances

    interaction_lf_acc = interaction_lf_correct_total / num_interactions

    return jsonf_data,sketch_acc, lf_acc, interaction_lf_acc


def load_data_new(sql_path, table_data, use_small=False):
    sql_data = []

    print("Loading data from %s" % sql_path)
    with open(sql_path) as inf:
        json_f = json.load(inf)

        data = lower_keys(json_f)

        sql_data += data

    table_data_new = {table['db_id']: table for table in table_data}

    if use_small:
        return sql_data[:80], table_data_new
    else:
        return sql_data, table_data_new


def load_dataset(dataset_dir, use_small=False, mode='train'):
    print("Loading from datasets...")

    TABLE_PATH = os.path.join(dataset_dir, "tables.json")
    TRAIN_PATH = os.path.join(dataset_dir, "pre_train.json")
    DEV_PATH = os.path.join(dataset_dir, "pre_dev.json")
    with open(TABLE_PATH) as inf:
        print("Loading data from %s"%TABLE_PATH)
        table_data = json.load(inf)
    if mode == 'train':
        train_sql_data, train_table_data = load_data_new(TRAIN_PATH, table_data, use_small=use_small)

        val_sql_data, val_table_data = load_data_new(DEV_PATH, table_data, use_small=use_small)
        return train_sql_data, train_table_data, val_sql_data, val_table_data
    else:
        val_sql_data, val_table_data = load_data_new(DEV_PATH, table_data, use_small=use_small)
        return val_sql_data, val_table_data

    #TODO 166
    # print(len(train_table_data.keys()))


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(model, checkpoint_name):
    torch.save(model.state_dict(), checkpoint_name)
def save_ckpt(state, is_best, current_w_name,best_w_name,current_w_name2):

    torch.save(state, current_w_name)
    shutil.copyfile(current_w_name, current_w_name2)
    if is_best:
        shutil.copyfile(current_w_name, best_w_name)


def save_args(args, path):
    with open(path, 'w') as f:
        f.write(json.dumps(vars(args), indent=4))

def init_log_checkpoint_path(args):
    dir_name = args.save
    save_path = os.path.join(os.path.curdir, dir_name)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    return save_path

def predicted_path(args):
    save_path = args.predict_save
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    return save_path
