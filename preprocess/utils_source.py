# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# -*- coding: utf-8 -*-
"""
# @Time    : 2019/12/09
# @Author  : xiaxia.wang
# @File    : utils.py
# @Software: PyCharm
"""
import os
import json
from nltk.stem import WordNetLemmatizer

VALUE_FILTER = ['what', 'how', 'list', 'give', 'show', 'find', 'id', 'order', 'when']
AGG = ['average', 'sum', 'max', 'min', 'minimum', 'maximum', 'between']

wordnet_lemmatizer = WordNetLemmatizer()


def generate_column_names_embedder_input(column_names, column_set, table_idxes, table_names):

    '''
    :param column_names:
    :param column_set:
    :param table_idxes:
    :param table_names:
            interaction['names'],interaction['col_set'],interaction['col_table'],interaction['table_names']
    :return:
    '''
    column_names_embedder_input = []
    column_names_embedder_input_idxes = []
    # print(column_names)
    # col_set = set()
    # for i in column_names:
    #     col_set.add(i)

    for i, (table_id, column_name) in enumerate(zip(table_idxes, column_names)):
        if table_id >= 0:
            table_name = table_names[table_id]
            column_name_embedder_input = table_name + ' . ' + column_name
            column_name_embedder_input_idx = (table_id, column_set.index(column_name))
        else:
            # TODO *
            column_name_embedder_input = column_name
            column_name_embedder_input_idx = (-1, column_set.index(column_name))

        column_names_embedder_input.append(column_name_embedder_input)
        column_names_embedder_input_idxes.append(column_name_embedder_input_idx)
        # column_names_embedder_input_to_id[column_name_embedder_input] = len(self.column_names_embedder_input) - 1

    for i, table_name in enumerate(table_names):
        column_name_embedder_input = table_name + ' . *'
        column_names_embedder_input.append(column_name_embedder_input)
        column_name_embedder_input_idx = (i, column_set.index('*'))
        column_names_embedder_input_idxes.append(column_name_embedder_input_idx)

    # for i, column_name in enumerate(column_set):
    #     if column_name != '*':
    #         column_name_embedder_input = column_name
    #         column_name_embedder_input_idx = (-1, i)
    #         column_names_embedder_input.append(column_name_embedder_input)
    #         column_names_embedder_input_idxes.append(column_name_embedder_input_idx)

    return column_names_embedder_input, column_names_embedder_input_idxes

def load_dataSets(args):
    with open(args.table_path, 'r', encoding='utf8') as f:
        table_datas = json.load(f)
    with open(args.data_path, 'r', encoding='utf8') as f:
        datas = json.load(f)

    output_tab = {}
    tables = {}
    tabel_name = set()
    for i in range(len(table_datas)):
        # TODO {'primary_keys': [1, 9],
        #  'column_names_original': [[-1, '*'], [0, 'Perpetrator_ID'], [0, 'People_ID'], [0, 'Date'], [0, 'Year'],
        #                            [0, 'Location'], [0, 'Country'], [0, 'Killed'], [0, 'Injured'], [1, 'People_ID'],
        #                            [1, 'Name'], [1, 'Height'], [1, 'Weight'], [1, 'Home Town']],
        #  'table_names': ['perpetrator', 'people'],
        #  'db_id': 'perpetrator',
        #  'foreign_keys': [[2, 9]],
        #  'column_names': [[-1, '*'], [0, 'perpetrator id'], [0, 'people id'], [0, 'date'], [0, 'year'], [0, 'location'],
        #                   [0, 'country'], [0, 'killed'], [0, 'injured'], [1, 'people id'], [1, 'name'], [1, 'height'],
        #                   [1, 'weight'], [1, 'home town']],
        #  'column_types': ['text', 'number', 'number', 'text', 'number', 'text', 'text', 'number', 'number', 'number',
        #                   'text', 'number', 'number', 'text'],
        #  'table_names_original': ['perpetrator', 'people']}
        table = table_datas[i]

        temp = {}
        temp['col_map'] = table['column_names']
        temp['table_names'] = table['table_names']
        tmp_col = []

        for cc in [x[1] for x in table['column_names']]:
            if cc not in tmp_col:
                tmp_col.append(cc)


        #TODO 去掉重复的列['*', 'perpetrator id', 'people id', 'date', 'year', 'location', 'country', 'killed', 'injured', 'name', 'height', 'weight', 'home town']
        table['col_set'] = tmp_col
        db_name = table['db_id']
        tabel_name.add(db_name)
        #TODO 未去掉重复
        table['schema_content'] = [col[1] for col in table['column_names']]
        # print(table['schema_content'])
        # print(table['col_set'])

        table['col_table'] = [col[0] for col in table['column_names']]

        column_names_embedder_input,column_names_embedder_input_idxes = generate_column_names_embedder_input(table['schema_content'], tmp_col, table['col_table'], table['table_names'])
        table['columns_names_embedder'] = column_names_embedder_input
        table['columns_names_embedder_idxes'] = column_names_embedder_input_idxes

        output_tab[db_name] = temp
        tables[db_name] = table

    for d in datas:
        #TODO Spider : dict_keys(['query_toks_no_value', 'query_toks', 'question', 'query', 'db_id', 'sql', 'question_toks'])

        # TODO Sparc : dict_keys(['interaction', 'final', 'database_id'])   interaction 里面 dict_keys(['utterance_toks', 'sql', 'query', 'utterance'])

        d['names'] = tables[d['database_id']]['schema_content']
        d['table_names'] = tables[d['database_id']]['table_names']
        d['col_set'] = tables[d['database_id']]['col_set']
        d['col_table'] = tables[d['database_id']]['col_table']
        d['columns_names_embedder'] = tables[d['database_id']]['columns_names_embedder']
        d['columns_names_embedder_idxes'] = tables[d['database_id']]['columns_names_embedder_idxes']
        keys = {}
        #TODO [[12, 7], [11, 1]]
        # print(tables[d['db_id']]['foreign_keys'])
        for kv in tables[d['database_id']]['foreign_keys']:
            keys[kv[0]] = kv[1]
            keys[kv[1]] = kv[0]
        #TODO {1: 11, 11: 1, 12: 7, 7: 12}
        # print(keys)
        #TODO [1, 7, 11]
        # print(tables[d['db_id']]['primary_keys'])
        for id_k in tables[d['database_id']]['primary_keys']:
            keys[id_k] = id_k
        #TODO {1: 1, 11: 11, 12: 7, 7: 7}  把外键覆盖了？？？？？？
        d['keys'] = keys

    return datas, tables

def group_header(toks, idx, num_toks, header_toks):
    for endIdx in reversed(range(idx + 1, num_toks+1)):
        sub_toks = toks[idx: endIdx]
        sub_toks = " ".join(sub_toks)
        if sub_toks in header_toks:
            return endIdx, sub_toks
    return idx, None

def fully_part_header(toks, idx, num_toks, header_toks):

    for endIdx in reversed(range(idx + 1, num_toks+1)):

        sub_toks = toks[idx: endIdx]

        if len(sub_toks) > 1:
            sub_toks = " ".join(sub_toks)
            if sub_toks in header_toks:
                return endIdx, sub_toks

    return idx, None

def partial_header(toks, idx, header_toks):
    def check_in(list_one, list_two):
        if len(set(list_one) & set(list_two)) == len(list_one) and (len(list_two) <= 3):
            return True
    for endIdx in reversed(range(idx + 1, len(toks))):
        sub_toks = toks[idx: min(endIdx, len(toks))]
        if len(sub_toks) > 1:
            flag_count = 0
            tmp_heads = None
            for heads in header_toks:
                if check_in(sub_toks, heads):
                    flag_count += 1
                    tmp_heads = heads
            if flag_count == 1:
                return endIdx, tmp_heads
    return idx, None

def symbol_filter(questions):
    question_tmp_q = []
    last_tok = None
    for q_id, q_val in enumerate(questions):

        if len(q_val) > 2 and q_val[0] in ["'", '"', '`', '鈥�', '鈥�','“','‘'] and q_val[-1] in ["'", '"', '`', '鈥�',"”",'’']:
            question_tmp_q.append("'")
            question_tmp_q += ["".join(q_val[1:-1])]
            question_tmp_q.append("'")
            last_tok = q_val[1:-1]
        elif len(q_val) > 2 and q_val[0] in ["'", '"', '`', '鈥�','“','‘'] :
            question_tmp_q.append("'")
            question_tmp_q += ["".join(q_val[1:])]
            last_tok = q_val[1:]
        elif len(q_val) > 2 and q_val[-1] in ["'", '"', '`', '鈥�',"”",'’']:
            question_tmp_q += ["".join(q_val[0:-1])]
            question_tmp_q.append("'")
            last_tok = q_val[0:-1]
        elif q_val in ["'", '"', '`', '鈥�', '鈥�', '``', "''"]:
            if last_tok is not None and len(last_tok)>1 and last_tok[0] in ["'", '"', '`', '鈥�', '鈥�', '``', "''"]:
                # print(last_tok, question_tmp_q)
                v = question_tmp_q.pop()
                question_tmp_q += ["'"]
                question_tmp_q += ["".join(v[1:])]

            # print('Q val : ',q_val)
            # print('Question : ',questions)
            # print('before: ', question_tmp_q)
            question_tmp_q += ["'"]
            # print('after : ', question_tmp_q)
            # print()
        else:
            question_tmp_q += [q_val]
            last_tok = q_val


    return question_tmp_q


def group_values(toks, idx, num_toks):
    def check_isupper(tok_lists):
        for tok_one in tok_lists:
            if tok_one[0].isupper() is False:
                return False
        return True

    for endIdx in reversed(range(idx + 1, num_toks + 1)):
        sub_toks = toks[idx: endIdx]

        if len(sub_toks) > 1 and check_isupper(sub_toks) is True:
            return endIdx, sub_toks
        if len(sub_toks) == 1:
            if sub_toks[0][0].isupper() and sub_toks[0].lower() not in VALUE_FILTER and \
                            sub_toks[0].lower().isalnum() is True:
                return endIdx, sub_toks
    return idx, None


def group_digital(toks, idx):
    test = toks[idx].replace(':', '')
    test = test.replace('.', '')
    if test.isdigit():
        return True
    else:
        return False

def group_symbol(toks, idx, num_toks):
    if toks[idx-1] == "'":
        for i in range(0, min(3, num_toks-idx)):
            if toks[i + idx] == "'":
                return i + idx, toks[idx:i+idx]
    return idx, None


def num2year(tok):
    if len(str(tok)) == 4 and str(tok).isdigit() and int(str(tok)[:2]) < 22 and int(str(tok)[:2]) > 15:
        return True
    return False

def set_header(toks, header_toks, tok_concol, idx, num_toks):
    def check_in(list_one, list_two):
        if set(list_one) == set(list_two):
            return True
    for endIdx in range(idx, num_toks):
        toks += tok_concol[endIdx]
        if len(tok_concol[endIdx]) > 1:
            break
        for heads in header_toks:
            if check_in(toks, heads):
                return heads
    return None
