# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# -*- coding: utf-8 -*-
"""
# @Time    : 2019/12/09
# @Author  : xiaxia.wang
# @File    : data_process.py
# @Software: PyCharm
"""
import json
import argparse
import nltk
import os
import pickle
import sys, logger
sys.path.insert(0,os.getcwd())
from preprocess.utils_source import symbol_filter, fully_part_header, group_header, partial_header, num2year, group_symbol, group_values, group_digital
from preprocess.utils_source import AGG, wordnet_lemmatizer
from preprocess.utils_source import load_dataSets

log_dir = 'log_tmp'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
record = logger.Logger(os.path.join(log_dir, os.path.basename(__file__).split('.')[0] + '.log'), 'w')
en_digit = {'two':2, 'three':3, 'four':4,
            'five':5, 'six':6, 'seven':7, 'eight':8,
            'nine':9, 'ten':10}

def judge_is_exists_digit(question_toks, tag_list):
    super_list = []
    digit_list = []
    # en_digit_list = []

    for i, (tok, tag) in enumerate(zip(question_toks, tag_list)):
        if tag == 'JJS' or tag == 'RBS' or \
                tok.endswith('est') or tok == 'most' or tok == 'least': # and (tag == 'JJ' or tag == 'RB')
            super_list.append(i)
        if tok.isdigit():
            digit_list.append(i)
        # if tok in en_digit.keys():
        #     en_digit_list.append(i)

    if super_list!=[] and digit_list!=[]:
        try: # assert len(digit_list) == 1
            for x in super_list:
                for y in digit_list:
                    if abs(x - y) == 1:
                        return True
        except Exception as e:
            return False
    return False

def is_en_digit(tok):
    #13,000
    res = []
    if tok == ',':
        return False,tok
    for i in tok:
        if not(i.isdigit() or i=='.' or i == ','):
            return False,''
        if i!=',':
            res.append(i)

    return True,''.join(res)

def convert_en_digit(question_toks, origin_question_toks):
    en_convert_question = []
    new_origin = []
    for i, tok in enumerate(question_toks):
        flag,res = is_en_digit(tok)

        if tok in en_digit.keys():
            en_convert_question.append(str(en_digit[tok]))
            new_origin.append(str(en_digit[tok]))
        elif flag:
            en_convert_question.append(res)
            new_origin.append(res)
        else:
            en_convert_question.append(tok)
            new_origin.append(origin_question_toks[i])
    return en_convert_question, new_origin

def add_common_sense(question_toks, origin_question_toks):
    new_question = []
    new_origin_question = []
    for i, (tok, origin_tok) in enumerate(zip(question_toks, origin_question_toks)):

       if tok.lower() == 'male':

            new_question.append(tok)
            new_question.append('m')
            new_origin_question.append(origin_tok)
            new_origin_question.append('M')
            continue
       if tok.lower() == 'female':
           new_question.append(tok)
           new_question.append('f')
           new_origin_question.append(origin_tok)
           new_origin_question.append('F')
           continue
       new_question.append(tok)
       new_origin_question.append(origin_tok)
    return new_question, new_origin_question

def longestCommonPrefix(strs):
    """
    :type strs: List[str]
    :rtype: str
    """
    if not strs:
        return ''
    s1 = min(strs)
    s2 = max(strs)
    for i, c in enumerate(s1):
        if c != s2[i]:
            return s1[:i]
    return s1

def convert_time_format(question_toks, origin_question_toks):

    TIME_MAP = {'January':1,'February':2,'March':3,"April":4,'May':5,
                'June':6,'July':7,'August':8,'September':9,
                'October':10,'November':11,'December':12}
    extra_time_map = {k.lower() : v for k,v in TIME_MAP.items()}

    for mon in list(TIME_MAP.keys()):
        extra_time_map[mon[:3].lower()] = TIME_MAP[mon]
    extra_time_map['Sept'.lower()] = 9

    try:
        time_list = []
        i = 0
        DAY = False
        YEAR = False
        ALL_FINISH = False
        start, end = -1, -1
        while i < len(question_toks):
            if question_toks[i].lower() in extra_time_map.keys():
                time_list.append(extra_time_map[question_toks[i].lower()])
                DAY = True
                start = i
                i += 1
                continue
            if DAY and (question_toks[i].isdigit() or question_toks[i].endswith('th')):
                if question_toks[i].isdigit():
                    time_list.append(int(question_toks[i]))
                else:
                    time_list.append(int(question_toks[i][:-2]))
                YEAR = True
                DAY = False
                end = i
                i += 1
                continue
            if YEAR:
                if question_toks[i].isdigit():
                    time_list.append(int(question_toks[i]))
                    end = i
                    ALL_FINISH = True
            if ALL_FINISH:
                break
            i += 1
        if len(time_list) == 3:
            time_format = ['%d-%02d-%02d'%(time_list[2],time_list[0],time_list[1])]
        elif len(time_list) == 2:
            time_format = ['%02d-%02d'%(time_list[0], time_list[1])]


        new_question = []
        new_ori = []
        COUNT = 0
        for i,(tok,ori) in enumerate(zip(question_toks,origin_question_toks)):
            if i>=start and i<=end:
                if COUNT == 0:
                    new_question.extend(time_format)
                    new_ori.extend(time_format)
                COUNT += 1
            else:
                new_ori.append(ori)
                new_question.append(tok)

    except Exception as e:
        new_question = question_toks
        new_ori = origin_question_toks

    return new_question, new_ori


def check_has_super_and_verify(question_toks, origin_question_toks):
    # print(question_toks)
    pos_tag_res = nltk.pos_tag(question_toks)
    question = [x[0] for x in pos_tag_res]
    tag_list = [x[1] for x in pos_tag_res]
    assert question_toks == question
    # record.put('before : ' + str(question_toks))
    # print('aaa',question_toks)
    question_toks, origin_question_toks = convert_en_digit(question_toks, origin_question_toks)
    # print('bbb',question_toks)
    question_toks, origin_question_toks = convert_time_format(question_toks, origin_question_toks)
    question_toks, origin_question_toks = add_common_sense(question_toks, origin_question_toks)
    # record.put('before : ' + str(question_toks) + '\n')

    if judge_is_exists_digit(question_toks, tag_list):

        return question_toks, origin_question_toks

    new_question = []
    new_origin_question = []
    flag = False
    for i, (tok,tag,origin_tok) in enumerate(zip(question_toks,tag_list, origin_question_toks)):

        if tag == 'JJS' or tag == 'RBS' or \
                tok.endswith('est') or tok == 'most' or tok == 'least':
            new_question.append(tok)
            new_question.append('1')
            new_origin_question.append(origin_tok)
            new_origin_question.append('1')
            flag = True
            continue
        new_question.append(tok)
        new_origin_question.append(origin_tok)

    has_digit = False
    for tok in new_question:
        if tok.isdigit():
            has_digit = True
            break
    if has_digit == False:

        new_question.append('1')
        new_origin_question.append('1')

    return new_question, new_origin_question

def process_datas(datas, args):
    """

    :param datas:
    :param args:
    :return:
    """
    with open(os.path.join(args.conceptNet, 'english_RelatedTo.pkl'), 'rb') as f:
        english_RelatedTo = pickle.load(f)

    with open(os.path.join(args.conceptNet, 'english_IsA.pkl'), 'rb') as f:
        english_IsA = pickle.load(f)

    # copy of the origin question_toks
    for d in datas:
        for inter in d['interaction']:
            if 'origin_utterance_toks' not in inter:
                inter['origin_utterance_toks'] = inter['utterance_toks']


    for entry in datas:

        table_names = []

        # TODO 表名 复数->单数 ['airlines', 'airports', 'flights']  ->  ['airline', 'airport', 'flight']

        for y in entry['table_names']:
            x = [wordnet_lemmatizer.lemmatize(x.lower()) for x in y.split(' ')]
            table_names.append(" ".join(x))

        header_toks = []
        header_toks_list = []

        # TODO 复数->单数   ；  每个列都是list
        # ['*', 'book club id', 'year', 'author or editor', ...]   ->  [['*'], ['book', 'club', 'id'], ['year'], ['author', 'or', 'editor'], ...]

        for y in entry['col_set']:
            x = [wordnet_lemmatizer.lemmatize(x.lower()) for x in y.split(' ')]
            header_toks.append(" ".join(x))
            header_toks_list.append(x)

        last_utter = None

        for utter_item in entry['interaction']:

            # record.put('Before : ' + str(utter_item['utterance_toks']))
            utter_item['utterance_toks'] = symbol_filter(utter_item['utterance_toks'])
            # record.put('After  : ' + str(utter_item['utterance_toks']))

            #TODO 把引号单独拿出来['What', 'is', 'the', 'location', 'of', 'the', 'bridge', 'named', "'", 'Kolob', 'Arch', "'", 'or', "'", 'Rainbow', 'Bridge', "'", '?']
            # print(entry['question_toks'])

            origin_question_toks = symbol_filter([x for x in utter_item['origin_utterance_toks'] if x.lower() != 'the'])

            question_toks = [wordnet_lemmatizer.lemmatize(x.lower()) for x in utter_item['utterance_toks'] if x.lower() != 'the']
            
            tmp_question = []
            for x in question_toks:
                if x == 'ha':
                    tmp_question.append('has')
                elif x == 'doe':
                    tmp_question.append('does')
                else:
                    tmp_question.append(x)
            question_toks = tmp_question

            '''
            优先从 turn (i,i-1,i-2,...1)开始
            进行 根据历史last_utter,值部分的匹配(针对数值部分，需要generate)
                文字部分，值的位置根据 Turn, valueS, Distance 定位
            '''
            print_info = False
            # if ' '.join(question_toks) == 'which department has most employee ? give me department name .':
            #     print(question_toks)
            #     pos_tag_res = nltk.pos_tag(question_toks)
            #     print(pos_tag_res)
            #     print_info = True
            # print('B : ', question_toks)
            question_toks, origin_question_toks = check_has_super_and_verify(question_toks,origin_question_toks)
            # print('A : ', question_toks)
            # print()
            assert len(question_toks) == len(origin_question_toks)


            utter_item['utterance_toks'] = question_toks

            num_toks = len(question_toks)
            idx = 0
            tok_concol = []
            origin_tok_concol = []
            type_concol = []
            #TODO ['what', 'are', 'all', 'airline', '?'] ->
            # [('what', 'WDT'), ('are', 'VBP'), ('all', 'DT'), ('airline', 'NN'), ('?', '.')]
            # print(question_toks)
            nltk_result = nltk.pos_tag(question_toks)

            while idx < num_toks:

                #TODO 在question中 找 匹配列名的至少大于1
                # fully header 完全匹配列名
                end_idx, header = fully_part_header(question_toks, idx, num_toks, header_toks)
                if header:
                    tok_concol.append(question_toks[idx: end_idx])

                    origin_tok_concol.append(origin_question_toks[idx : end_idx])

                    type_concol.append(["col"])
                    idx = end_idx
                    continue
                
                # TODO 在question中 找 匹配列名的为1
                # check for column
                end_idx, header = group_header(question_toks, idx, num_toks, header_toks)
                if header:
                    tok_concol.append(question_toks[idx: end_idx])
                    origin_tok_concol.append(origin_question_toks[idx: end_idx])
                    type_concol.append(["col"])
                    idx = end_idx
                    continue
                
                # TODO 在question中 找 匹配的表名
                # check for table
                end_idx, tname = group_header(question_toks, idx, num_toks, table_names)

                if tname:
                    tok_concol.append(question_toks[idx: end_idx])
                    origin_tok_concol.append(origin_question_toks[idx: end_idx])
                    type_concol.append(["table"])
                    idx = end_idx
                    continue

                # check for partial column

                end_idx, tname = partial_header(question_toks, idx, header_toks_list)
                if tname:

                    tok_concol.append(tname)
                    origin_tok_concol.append(origin_question_toks[idx: end_idx])
                    type_concol.append(["col"])
                    idx = end_idx
                    continue

                # check for aggregation
                end_idx, agg = group_header(question_toks, idx, num_toks, AGG)
                if agg:

                    tok_concol.append(question_toks[idx: end_idx])
                    origin_tok_concol.append(origin_question_toks[idx: end_idx])
                    type_concol.append(["agg"])
                    idx = end_idx
                    continue

                if nltk_result[idx][1] == 'RBR' or nltk_result[idx][1] == 'JJR':
                    tok_concol.append([question_toks[idx]])
                    origin_tok_concol.append([origin_question_toks[idx]])
                    type_concol.append(['MORE'])
                    idx += 1
                    continue

                if nltk_result[idx][1] == 'RBS' or nltk_result[idx][1] == 'JJS':
                    tok_concol.append([question_toks[idx]])
                    origin_tok_concol.append([origin_question_toks[idx]])
                    type_concol.append(['MOST'])
                    idx += 1
                    continue

                # # string match for Time Format
                # if num2year(question_toks[idx]):
                #     question_toks[idx] = 'year'
                #     end_idx, header = group_header(question_toks, idx, num_toks, header_toks)
                #     if header:
                #         tok_concol.append(question_toks[idx: end_idx])
                #         type_concol.append(["col"])
                #         idx = end_idx
                #         continue

                def get_concept_result(toks, graph):
                    for begin_id in range(0, len(toks)):
                        for r_ind in reversed(range(1, len(toks) + 1 - begin_id)):
                            tmp_query = "_".join(toks[begin_id:r_ind])
                            if tmp_query in graph:
                                mi = graph[tmp_query]
                                for col in entry['col_set']:
                                    if col in mi:
                                        return col

                end_idx, symbol = group_symbol(question_toks, idx, num_toks)

                if symbol:

                    tmp_toks = [x for x in question_toks[idx: end_idx]]

                    assert len(tmp_toks) > 0, print(symbol, question_toks)
                    pro_result = get_concept_result(tmp_toks, english_IsA)
                    if pro_result is None:
                        pro_result = get_concept_result(tmp_toks, english_RelatedTo)
                    if pro_result is None:
                        pro_result = "NONE"
                    for tmp in tmp_toks:
                        tok_concol.append([tmp])
                        type_concol.append([pro_result])
                        pro_result = "NONE"

                    origin_tmp_toks = [x for x in origin_question_toks[idx: end_idx]]
                    for tmp in origin_tmp_toks:
                        origin_tok_concol.append([tmp])
                    idx = end_idx
                    continue

                try:
                    end_idx, values = group_values(origin_question_toks, idx, num_toks)

                    if values and (len(values) > 1 or question_toks[idx - 1] not in ['?', '.']):

                        tmp_toks = [wordnet_lemmatizer.lemmatize(x) for x in question_toks[idx: end_idx] if x.isalnum() is True]

                        assert len(tmp_toks) > 0, print(question_toks[idx: end_idx], values, question_toks, idx, end_idx)
                        pro_result = get_concept_result(tmp_toks, english_IsA)
                        if pro_result is None:
                            pro_result = get_concept_result(tmp_toks, english_RelatedTo)
                        if pro_result is None:
                            pro_result = "NONE"
                        for tmp in tmp_toks:
                            tok_concol.append([tmp])
                            type_concol.append([pro_result])
                            pro_result = "NONE"
                        origin_tmp_toks = [x for x in origin_question_toks[idx: end_idx] if x.isalnum() is True]
                        for tmp in origin_tmp_toks:
                            origin_tok_concol.append([tmp])
                        idx = end_idx
                        continue
                except Exception as e:
                    print(question_toks)
                    print('Group Value Error')
                result = group_digital(question_toks, idx)
                if result is True:
                    tok_concol.append(question_toks[idx: idx + 1])
                    origin_tok_concol.append(origin_question_toks[idx: idx + 1])
                    type_concol.append(["value"])
                    idx += 1
                    continue
                if question_toks[idx] == ['ha']:
                    question_toks[idx] = ['have']

                tok_concol.append([question_toks[idx]])
                origin_tok_concol.append([origin_question_toks[idx]])
                type_concol.append(['NONE'])
                idx += 1
                continue

            utter_item['utterance_arg'] = tok_concol
            utter_item['utterance_arg_type'] = type_concol
            utter_item['nltk_pos'] = nltk_result
            utter_item['origin_utterance_arg'] = origin_tok_concol
            assert len(tok_concol) == len(origin_tok_concol)

            # print(utter_item['utterance_toks'])
            # print(utter_item['utterance_arg'])
            # print()

    return datas


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, help='dataset', required=True)
    arg_parser.add_argument('--table_path', type=str, help='table dataset', required=True)
    arg_parser.add_argument('--output', type=str, help='output data')
    args = arg_parser.parse_args()
    args.conceptNet = 'preprocess/conceptNet'

    # python3 data_process_sparc.py --data_path ../sparc/dev_no_value.json --table_path ../sparc/tables.json --output ../schema_linking_process/dev_no_value_schema_linking.json
    # loading dataSets
    datas, table = load_dataSets(args)

    # process datasets
    process_result = process_datas(datas, args)

    with open(args.output, 'w') as f:
        json.dump(datas, f)
    print('process data finish!!!')


