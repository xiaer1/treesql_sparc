# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# -*- coding: utf-8 -*-
"""
# @Time    : 2019/12/09
# @Author  : xiaxia.wang
# @File    : data_process.py
# @Software: PyCharm
"""

import argparse
import json
import sys,os
sys.path.insert(0,os.getcwd())
import copy, logger
from utils_source import load_dataSets
from preprocess.utils_source import wordnet_lemmatizer, symbol_filter

sys.path.append(os.getcwd())

limit_value_match = 0
filter_value_match = 0
filter_value_part_match = 0
filter_between_value_match = 0
filter_column = 0
from src.rule.semQL import Root1, Root, N, A, Turn, ValueS, Distance,  Sel, Sup, Filter, Order,TC,action_map,inv_action_map


log_dir = 'log_tmp'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
record = logger.Logger(os.path.join(log_dir, os.path.basename(__file__).split('.')[0] + '.log'), 'w')

record_error = logger.Logger(os.path.join(log_dir, os.path.basename(__file__).split('.')[0] + '_error.log'), 'w')

class Parser:
    def __init__(self):
        self.copy_selec = None
        self.sel_result = []
        self.colSet = set()

    def _init_rule(self):
        self.copy_selec = None
        self.colSet = set()

    def _parse_root(self, sql,parent):
        """
        parsing the sql by the grammar
        R ::= Select | Select Filter | Select Order | ... |
        :return: [R(), states]
        """
        use_sup, use_ord, use_fil = True, True, False

        if sql['sql']['limit'] == None:
            use_sup = False

        if sql['sql']['orderBy'] == []:
            use_ord = False
        elif sql['sql']['limit'] != None:
            use_ord = False

        # check the where and having
        if sql['sql']['where'] != [] or \
                        sql['sql']['having'] != []:
            use_fil = True

        #TODO 因为用栈，所以这里倒序 ， 实际上先用 sel,sup,filter
        
        if use_fil and use_sup:
            return [Root(0,parent)], ['FILTER', 'SUP', 'SEL']
        elif use_fil and use_ord:
            return [Root(1,parent)], ['ORDER', 'FILTER', 'SEL']
        elif use_sup:
            return [Root(2,parent)], ['SUP', 'SEL']
        elif use_fil:
            return [Root(3,parent)], ['FILTER', 'SEL']
        elif use_ord:
            return [Root(4,parent)], ['ORDER', 'SEL']
        else:
            return [Root(5,parent)], ['SEL']

    def _parser_column0(self, sql, select):
        """
        Find table of column '*'
        :return: T(table_id)
        """
        #TODO dict_keys(['groupBy', 'having', 'where', 'intersect', 'limit', 'union', 'select', 'orderBy', 'from', 'except'])
        # print(sql['sql'].keys())

        if len(sql['sql']['from']['table_units']) == 1:
            if sql['sql']['from']['table_units'][0][0] == 'table_unit':
                return sql['sql']['from']['table_units'][0][1]
            else:
                #TODO
                # dict_keys(['groupBy', 'having', 'where', 'intersect', 'limit', 'union', 'select', 'orderBy', 'from', 'except'])
                # print(sql['sql']['from']['table_units'][0][1].keys())
                # print('haha->')
                # print(sql['query'])

                return -1
        else:

            #TODO 多表
            table_list = []
            for tmp_t in sql['sql']['from']['table_units']:
                if type(tmp_t[1]) == int:
                    table_list.append(tmp_t[1])


            table_set, other_set = set(table_list), set()

            for sel_p in select:
                if sel_p[1][1][1] != 0:
                    other_set.add(sql['col_table'][sel_p[1][1][1]])

            #TODO ？？？？？？为什么1,3,5
            if len(sql['sql']['where']) == 1:
                other_set.add(sql['col_table'][sql['sql']['where'][0][2][1][1]])

            elif len(sql['sql']['where']) == 3:
                other_set.add(sql['col_table'][sql['sql']['where'][0][2][1][1]])
                other_set.add(sql['col_table'][sql['sql']['where'][2][2][1][1]])
            elif len(sql['sql']['where']) == 5:
                other_set.add(sql['col_table'][sql['sql']['where'][0][2][1][1]])
                other_set.add(sql['col_table'][sql['sql']['where'][2][2][1][1]])
                other_set.add(sql['col_table'][sql['sql']['where'][4][2][1][1]])
            table_set = table_set - other_set
            if len(table_set) == 1:
                return list(table_set)[0]
            elif len(table_set) == 0 and sql['sql']['groupBy'] != []:
                return sql['col_table'][sql['sql']['groupBy'][0][1]]
            else:
                global error_cnt
                question = sql['question']
                self.sel_result.append(question)
                error_cnt+=1
                #TODO 随机挑选一个，在以下情况是 正确的：
                # SELECT * FROM INVESTORS AS T1 JOIN TRANSACTIONS AS T2 ON T1.investor_id  =  T2.investor_id
                # 虽然指明给 table INVESTORS.*, 但是反解析时，会将 表名去掉 => select *
                return sql['sql']['from']['table_units'][0][1]

    def _parse_select(self, sql,parent,history_utterance):
        """
        parsing the sql by the grammar
        Select ::= A | AA | AAA | ... |
        A ::= agg column table
        :return: [Sel(), states]
        """
        result = []

        #TODO [[3, [0, [0, 0, False], None]]]   [[ 聚合,[0,[0,列id,False,None]]]
        select = sql['sql']['select'][1]

        result.append(Sel(0,parent))
        sub_parent = action_map[Sel]
        result.append(N(len(select) - 1,sub_parent))

        #TODO select [   [0, [0, [0, 2, False], None]],  [0, [0, [0, 6, False], None]]    ]

        for sel in select:
            #TODO sel : [0, [0, [0, 2, False], None]]   sel[1][1][1] == 2
            #TODO 保存 第一个是聚合函数id,A(none)
            result.append(A(sel[0],sub_parent))
            #TODO 保存列名

            self.colSet.add(sql['col_set'].index(sql['names'][sel[1][1][1]]))

            col_id = sql['col_set'].index(sql['names'][sel[1][1][1]])

            # now check for the situation with *
            #TODO 保存表名 ， 特殊形似，如果是select * ， 那么从sql语句进行推理出属于哪个表。（比较复杂）
            if sel[1][1][1] == 0:
                table_id = self._parser_column0(sql, select)
                if table_id == -1:
                    col_id = sql['col_set'].index('*')
                tc_idx = sql['columns_names_embedder_idxes'].index((table_id,col_id))
                result.append(TC(tc_idx,sub_parent))

                #TODO 类似 SELECT COUNT(*) FROM (SELECT COUNT(*) ... )形式 ，from不是表格
                if table_id == -1:
                    global from_T
                    full_query={}
                    query = {}


                    full_query['names'] = sql['names']
                    full_query['col_table'] = sql['col_table']
                    full_query['col_set'] = sql['col_set']
                    full_query['table_names'] = sql['table_names']
                    full_query['keys'] = sql['keys']
                    full_query['columns_names_embedder'] = sql['columns_names_embedder']
                    full_query['columns_names_embedder_idxes'] = sql['columns_names_embedder_idxes']
                    full_query['nltk_pos'] = sql['nltk_pos']
                    full_query['utterance_arg_type'] = sql['utterance_arg_type']
                    full_query['utterance_arg'] = sql['utterance_arg']
                    # print(full_query.keys())
                    # print(sql.keys())
                    query['utterance_toks'] = sql['utterance_toks']
                    query['sql'] = sql['sql']['from']['table_units'][0][1]
                    # query['query_toks_no_value'] = sql['query_toks_no_value']
                    query['query'] = sql['query']
                    query['utterance_arg_type'] = sql['utterance_arg_type']
                    query['utterance_arg'] = sql['utterance_arg']

                    from_T += 1
                    from_result = self.full_parse(full_query,query, history_utterance)
                    result.extend(from_result)
            else:
                table_id = sql['col_table'][sel[1][1][1]]
                tc_idx = sql['columns_names_embedder_idxes'].index((table_id, col_id))
                result.append(TC(tc_idx,sub_parent))

            if not self.copy_selec:
                self.copy_selec = [copy.deepcopy(result[-2]), copy.deepcopy(result[-1])]



        return result, None

    def _parse_sup(self, sql, parent, history_utterance):
        """
        parsing the sql by the grammar
        Sup ::= Most A | Least A
        A ::= agg column table
        :return: [Sup(), states]
        """
        result = []
        select = sql['sql']['select'][1]
        # print(sql['sql'],sql)
        if sql['sql']['limit'] == None or sql['sql']['orderBy'] == []:
            return result, None
        #  ['Which', 'breed', 'codes', 'are', 'the', 'most', 'popular', 'two', '?'],
        #  'SELECT breed_code, count(*) FROM Dogs GROUP BY breed_code limit 2' 没法处理

        if sql['sql']['orderBy'][0] == 'desc':
            result.append(Sup(0,parent))
        else:
            result.append(Sup(1,parent))
        sub_parent = action_map[Sup]
        result.append(A(sql['sql']['orderBy'][1][0][1][0],sub_parent))
        self.colSet.add(sql['col_set'].index(sql['names'][sql['sql']['orderBy'][1][0][1][1]]))
        col_id = sql['col_set'].index(sql['names'][sql['sql']['orderBy'][1][0][1][1]])

        if sql['sql']['orderBy'][1][0][1][1] == 0:
            table_id = self._parser_column0(sql, select)
            if table_id == -1:
                col_id = sql['col_set'].index('*')
            tc_idx = sql['columns_names_embedder_idxes'].index((table_id, col_id))
            result.append(TC(tc_idx,sub_parent))
        else:
            table_id = sql['col_table'][sql['sql']['orderBy'][1][0][1][1]]
            tc_idx = sql['columns_names_embedder_idxes'].index((table_id, col_id))
            result.append(TC(tc_idx,sub_parent))

        global limit_value_match

        limit_value = str(sql['sql']['limit'])
        record.put('Value : ' + limit_value)
        for turn_idx, utter in enumerate(reversed(history_utterance)):
            t_idx = len(history_utterance) - 1 - turn_idx
            record.put('turn idx--->' + str(t_idx) + '\t' + str(utter['utterance_arg']))
            value_idx, dist = self._find_values_in_each_utterance(limit_value, utter['utterance_arg'])
            if value_idx != -1:
                record.put('\t Match\n\n')
                result.append(Turn(t_idx, sub_parent))
                result.append(ValueS(value_idx, sub_parent))
                assert dist == 0
                result.append(Distance(0, sub_parent))
                limit_value_match += 1
                break


        #Find max_score match in current turn
        if value_idx == -1:
            utter_arg = history_utterance[-1]['utterance_arg']
            max_match = dict()
            for i, tok in enumerate(utter_arg):
                if tok.isdigit():
                    max_match[i] = len(set(limit_value) & set(tok)) / len(set(tok))

            try:
                max_match = sorted(max_match.items(), key=lambda x:x[1], reverse=True)
                value_idx = max_match[0][0]

            except Exception as e:
                print('Limit Max Error')
                exit(1)
                value_idx = len(utter_arg) - 1
            result.append(Turn(len(history_utterance) - 1, sub_parent))
            result.append(ValueS(value_idx, sub_parent))
            result.append(Distance(0, sub_parent))


        return result, None

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            pass
        return False

    def _find_values_in_each_utterance(self, value, utter_arg):

        idx_list = []

        # mmm = 'A Love of a Lifetime'
        # utter_arg = [['tell'], ['me'], ['rating'], ['of'], ['episode'], ['titled'], ["'"],
        #              ['a'], ['love'], ['of'], ['a'], ['lifetime'], ["'"], ['.'], ['1']]

        v = symbol_filter([x for x in value.split(' ')])
        # if value == mmm:
        #     print(v)
        value_tokens = [wordnet_lemmatizer.lemmatize(x.lower().lstrip('%').rstrip('%')) for x in v if not( x=='\'' or x=='\"')]
        # if value == mmm:
        #     print(value_tokens)
        #     exit(1)
        # print(utter_arg)
        v_ind = 0
        correct = 0
        tmp_list = []
        for ind, q_token in enumerate(utter_arg):
            for each_token in q_token:

                if v_ind < len(value_tokens) and (each_token == value_tokens[v_ind] or (self.is_number(each_token) \
                                                                                        and self.is_number(
                            value_tokens[v_ind]) \
                                                                                        and float(each_token) == float(
                            value_tokens[v_ind]))):
                    v_ind += 1
                    correct += 1

                else:
                    correct = 0
                    v_ind = 0
                if len(tmp_list) == len(value_tokens):
                    break
            if correct != 0:
                # if len(tmp_list) >= 1 and ind - tmp_list[-1] == 1:  # 如果不连续相等，退出
                # break
                tmp_list.append(ind)

            else:
                if tmp_list != []:
                    idx_list.append(tmp_list)
                    tmp_list = []
                    v_ind = 0
        # vale = ''.join([''.join(x) for x in question_tokens])
        if tmp_list != []:
            idx_list.append(tmp_list)
            tmp_list = []

        if idx_list != []:
            idx_list = sorted(idx_list, key=lambda x: len(x), reverse=True)[0]

        value_idx = -1
        dist = 0
        if idx_list!=[]:
            value_idx = idx_list[0]
            dist = idx_list[-1] - idx_list[0]
        # print(value_idx)
        # exit(1)
        return value_idx, dist

    def _parse_filter(self, sql,parent, history_utterance):
        """
        parsing the sql by the grammar
        Filter ::= and Filter Filter | ... |
        A ::= agg column table
        :return: [Filter(), states]
        """
        result = []
        # check the where
        '''
        首先，where 和 having 同时不为空时，进入 Filter and Filter Filter，因为每部分都是个filter，会递归下去
        
        如果where不为空：
            
        
        '''
        if sql['sql']['where'] != [] and sql['sql']['having'] != []:
            result.append(Filter(0,parent))
        #TODO [[False, 3, [0, [0, 10, False], None], 1880.0, None]]   , len()==1
        # print(sql['sql']['where'])
        
        #TODO [[False, 5, [0, [3, 0, False], None], 2.0, None]]
        # print(sql['sql']['having'])
        if sql['sql']['where'] != []:
            # check the not and/or
            if len(sql['sql']['where']) == 1:
                result.extend(self.parse_one_condition(sql['sql']['where'][0], sql['names'], sql,parent, history_utterance))
            elif len(sql['sql']['where']) == 3:
                if sql['sql']['where'][1] == 'or':
                    result.append(Filter(1,parent))
                else:
                    result.append(Filter(0,parent))
                result.extend(self.parse_one_condition(sql['sql']['where'][0], sql['names'], sql,parent, history_utterance))
                result.extend(self.parse_one_condition(sql['sql']['where'][2], sql['names'], sql,parent, history_utterance))
            else:
                if sql['sql']['where'][1] == 'and' and sql['sql']['where'][3] == 'and':
                    result.append(Filter(0,parent))
                    result.extend(self.parse_one_condition(sql['sql']['where'][0], sql['names'], sql,parent, history_utterance))
                    result.append(Filter(0,parent))
                    result.extend(self.parse_one_condition(sql['sql']['where'][2], sql['names'], sql,parent, history_utterance))
                    result.extend(self.parse_one_condition(sql['sql']['where'][4], sql['names'], sql,parent, history_utterance))
                elif sql['sql']['where'][1] == 'and' and sql['sql']['where'][3] == 'or':
                    result.append(Filter(1,parent))
                    result.append(Filter(0,parent))
                    result.extend(self.parse_one_condition(sql['sql']['where'][0], sql['names'], sql,parent, history_utterance))
                    result.extend(self.parse_one_condition(sql['sql']['where'][2], sql['names'], sql,parent, history_utterance))
                    result.extend(self.parse_one_condition(sql['sql']['where'][4], sql['names'], sql,parent, history_utterance))
                elif sql['sql']['where'][1] == 'or' and sql['sql']['where'][3] == 'and':
                    result.append(Filter(1,parent))
                    result.append(Filter(0,parent))
                    result.extend(self.parse_one_condition(sql['sql']['where'][2], sql['names'], sql,parent, history_utterance))
                    result.extend(self.parse_one_condition(sql['sql']['where'][4], sql['names'], sql,parent,history_utterance))
                    result.extend(self.parse_one_condition(sql['sql']['where'][0], sql['names'], sql,parent,history_utterance))
                else:
                    result.append(Filter(1,parent))
                    result.append(Filter(1,parent))
                    result.extend(self.parse_one_condition(sql['sql']['where'][0], sql['names'], sql,parent,history_utterance))
                    result.extend(self.parse_one_condition(sql['sql']['where'][2], sql['names'], sql,parent,history_utterance))
                    result.extend(self.parse_one_condition(sql['sql']['where'][4], sql['names'], sql,parent,history_utterance))

        # check having
        if sql['sql']['having'] != []:
            result.extend(self.parse_one_condition(sql['sql']['having'][0], sql['names'], sql,parent,history_utterance))
        return result, None

    def _parse_order(self, sql,parent):
        """
        parsing the sql by the grammar
        Order ::= asc A | desc A
        A ::= agg column table
        :return: [Order(), states]
        """
        result = []
        sub_parent = action_map[Order]

        query_with_value = [wordnet_lemmatizer.lemmatize(x.lower()) for x in sql['query'].split(' ')]

        if 'order' not in query_with_value or 'by' not in query_with_value:
            return result, None
        elif 'limit' in query_with_value:
            return result, None
        else:
            if sql['sql']['orderBy'] == []:
                return result, None
            else:
                select = sql['sql']['select'][1]
                if sql['sql']['orderBy'][0] == 'desc':
                    result.append(Order(0,parent))
                else:
                    result.append(Order(1,parent))
                result.append(A(sql['sql']['orderBy'][1][0][1][0],sub_parent))
                self.colSet.add(sql['col_set'].index(sql['names'][sql['sql']['orderBy'][1][0][1][1]]))

                col_id = sql['col_set'].index(sql['names'][sql['sql']['orderBy'][1][0][1][1]])

                if sql['sql']['orderBy'][1][0][1][1] == 0:
                    table_id = self._parser_column0(sql, select)
                    if table_id == -1:
                        col_id = sql['col_set'].index('*')
                    tc_idx = sql['columns_names_embedder_idxes'].index((table_id, col_id))
                    result.append(TC(tc_idx,sub_parent))
                else:
                    table_id = sql['col_table'][sql['sql']['orderBy'][1][0][1][1]]
                    tc_idx = sql['columns_names_embedder_idxes'].index((table_id, col_id))
                    result.append(TC(tc_idx,sub_parent))
        return result, None

    def string_distance(self, str1, str2):
        """
        计算两个字符串之间的编辑距离
        @author: 仰起脸笑的像满月
        @date: 2019/05/15
        :param str1:
        :param str2:
        :return:
        """
        import numpy as np
        m = str1.__len__()
        n = str2.__len__()
        distance = np.zeros((m + 1, n + 1))

        for i in range(0, m + 1):
            distance[i, 0] = i
        for i in range(0, n + 1):
            distance[0, i] = i

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i - 1] == str2[j - 1]:
                    cost = 0
                else:
                    cost = 1
                distance[i, j] = min(distance[i - 1, j] + 1, distance[i, j - 1] + 1,
                                     distance[i - 1, j - 1] + cost)  # 分别对应删除、插入和替换

        return distance[m, n]

    def parse_one_condition(self, sql_condit, names, sql,parent, history_utterance):
        result = []
        # check if V(root)
        nest_query = True

        #TODO sql_condit : eg.[False, 3, [0, [0, 10, False], None], 1880.0, None]

        if type(sql_condit[3]) != dict:
            nest_query = False

        #TODO where 中 (not_op, op_id, val_unit, val1, val2)
        if sql_condit[0] == True:  #TODO not
            if sql_condit[1] == 9:  #TODO like
                # not like only with values
                #TODO Filter not_like
                fil = Filter(10,parent)
            elif sql_condit[1] == 8: #TODO in
                # not in with Root

                # TOOD Filter not_in A Root
                fil = Filter(19,parent)
            else:
                print(sql_condit[1])
                raise NotImplementedError("not implement for the others FIL")
        else: #TODO 没有not
            # check for Filter (<,=,>,!=,between, >=,  <=, ...)
            single_map = {1:8,2:2,3:5,4:4,5:7,6:6,7:3}
            nested_map = {1:15,2:11,3:13,4:12,5:16,6:17,7:14}
            if sql_condit[1] in [1, 2, 3, 4, 5, 6, 7]:
                if nest_query == False:
                    fil = Filter(single_map[sql_condit[1]],parent)
                else:
                    fil = Filter(nested_map[sql_condit[1]],parent)
            elif sql_condit[1] == 9:   # like
                fil = Filter(9,parent)
            elif sql_condit[1] == 8:   # in
                fil = Filter(18,parent)
            else:
                print(sql_condit[1])
                raise NotImplementedError("not implement for the others FIL")
        sub_parent = action_map[Filter]
        result.append(fil)
        result.append(A(sql_condit[2][1][0],sub_parent))
        self.colSet.add(sql['col_set'].index(sql['names'][sql_condit[2][1][1]]))
        col_id = sql['col_set'].index(sql['names'][sql_condit[2][1][1]])

        #TODO 特殊 * 列
        if sql_condit[2][1][1] == 0:
            select = sql['sql']['select'][1]
            table_id = self._parser_column0(sql, select)
            if table_id == -1:
                col_id = sql['col_set'].index('*')
            tc_idx = sql['columns_names_embedder_idxes'].index((table_id, col_id))
            result.append(TC(tc_idx,sub_parent))
        else:
            table_id = sql['col_table'][sql_condit[2][1][1]]
            tc_idx = sql['columns_names_embedder_idxes'].index((table_id, col_id))
            result.append(TC(tc_idx,sub_parent))

        global  filter_value_match, filter_value_part_match, filter_between_value_match,filter_column


        def get_value_info(value,history_utterance):
            global filter_value_match
            for turn_idx, utter in enumerate(reversed(history_utterance)):
                t_idx = len(history_utterance) - 1 - turn_idx
                value_idx, dist = self._find_values_in_each_utterance(value, utter['utterance_arg'])
                if value_idx != -1:

                    return t_idx, value_idx, dist,True
            # find max score
            if value_idx == -1:
                vt = symbol_filter([x for x in value.split(' ')])
                value_tokens = [wordnet_lemmatizer.lemmatize(x.lower().lstrip('%').rstrip('%')) for x in vt if not (x == '\'' or x == '\"' or x=="'" or x == '"')]

                value_info_dict = dict()
                for turn_idx, utter in enumerate(reversed(history_utterance)):

                    t_idx = len(history_utterance) - 1 - turn_idx


                    value_info_list = []

                    for v in value_tokens:
                        val_info = dict()
                        for ind, q_token in enumerate(utter['utterance_arg']):
                            q_v = ''.join(q_token)
                            score = self.string_distance(q_v, str(v))
                            # if score <= len(v)/2 :
                            val_info[ind] = (q_v,score)

                        if len(val_info)!=0:
                            val_info = sorted(val_info.items(), key=lambda x:x[1][-1])[0]
                            value_info_list.append(val_info)
                    if value_info_list != []:
                        sum_score = 0
                        for pp in value_info_list:
                            sum_score += pp[1][1]
                        value_info_dict[t_idx] = (sum_score, value_info_list)
                value_info_dict = sorted(value_info_dict.items(), key=lambda x:x[1][0])[0]
                rtn_turn = value_info_dict[0]
                match_value = value_info_dict[1][1]
                if len(match_value) == 1:
                    value_idx = match_value[0][0]
                    dist = 0
                else:
                    diff_list = []
                    for i, m in enumerate(match_value):
                        if i == 0:
                            diff_list.append((m[0],0))
                        else:
                            last_idx = diff_list[-1][0]
                            diff_list.append((m[0], m[0] - last_idx))
                    new_list = []
                    for idx, diff in diff_list:
                        if diff>=0 and diff<=2:
                            new_list.append(idx)
                        else:
                            new_list.pop()
                            new_list.append(idx)

                    value_idx = new_list[0]
                    dist = new_list[-1] - new_list[0]

                return rtn_turn,value_idx, dist ,False




        if type(sql_condit[3]) != dict:
            if sql_condit[4]!=None: #between 2 value
                value1 = str(sql_condit[3]).strip("'").strip('"')
                value2 = str(sql_condit[4]).strip("'").strip('"')
                t_idx, value_idx, dist, exact_match1 = get_value_info(value1, history_utterance)

                result.append(Turn(t_idx, sub_parent))
                result.append(ValueS(value_idx, sub_parent))
                result.append(Distance(dist, sub_parent))
                t_idx, value_idx, dist, exact_match2 = get_value_info(value2, history_utterance)
                if exact_match1 and exact_match2:
                    filter_between_value_match += 1
                result.append(Turn(t_idx, sub_parent))
                result.append(ValueS(value_idx, sub_parent))
                result.append(Distance(dist, sub_parent))

            elif isinstance(sql_condit[3], list):

                value = str(sql['names'][sql_condit[3][1]]).strip("'").strip('"')
                t_idx, value_idx, dist, exact_match = get_value_info(value, history_utterance)
                filter_column += 1
                # record_error.put('Value : ' + value)

                va = history_utterance[t_idx]['utterance_arg'][value_idx:value_idx+dist+1]
                # record_error.put('List : ' + str(va) + '\n'*2)
                result.append(Turn(t_idx, sub_parent))
                result.append(ValueS(value_idx, sub_parent))
                result.append(Distance(dist, sub_parent))

            else: # str
                value = str(sql_condit[3]).strip("'").strip('"')
                t_idx, value_idx, dist, exact_match = get_value_info(value, history_utterance)
                if exact_match:
                    filter_value_match += 1
                else:
                    filter_value_part_match += 1
                result.append(Turn(t_idx, sub_parent))
                result.append(ValueS(value_idx, sub_parent))
                result.append(Distance(dist, sub_parent))


        # check for the nested value
        if type(sql_condit[3]) == dict:
            nest_query = {}
            nest_query['names'] = names
            # nest_query['query_toks_no_value'] = ""
            nest_query['sql'] = sql_condit[3]
            nest_query['col_table'] = sql['col_table']
            nest_query['col_set'] = sql['col_set']
            nest_query['table_names'] = sql['table_names']
            nest_query['question'] = sql['question']
            nest_query['query'] = sql['query']
            nest_query['keys'] = sql['keys']
            nest_query['columns_names_embedder_idxes'] = sql['columns_names_embedder_idxes']
            nest_query['columns_names_embedder'] = sql['columns_names_embedder']
            nest_query['utterance_arg_type'] = sql['utterance_arg_type']
            nest_query['utterance_arg'] = sql['utterance_arg']

            result.extend(self.parser(nest_query, history_utterance))

        return result

    def _parse_step(self, state, sql, history_utterance):
        parent = action_map[Root1]
        if state == 'ROOT':
            return self._parse_root(sql,parent)
        parent = action_map[Root]
        if state == 'SEL':
            return self._parse_select(sql,parent, history_utterance)

        elif state == 'SUP':
            return self._parse_sup(sql,parent, history_utterance)

        elif state == 'FILTER':
            return self._parse_filter(sql,parent, history_utterance)

        elif state == 'ORDER':
            return self._parse_order(sql,parent)
        else:
            raise NotImplementedError("Not the right state")

    def full_parse(self, full_query, query, history_utterance):
        '''
        :param full_query: dict_keys(['names', 'keys', 'interaction', 'table_names', 'col_set', 'final', 'database_id', 'col_table',
                                    'columns_names_embedder', 'columns_names_embedder_idxes'])
        :param query:  dict_keys(['utterance', 'utterance_toks', 'query', 'query_toks_no_value', 'sql'])
        :return:
        '''

        # dict_keys(['groupBy', 'where', 'from', 'orderBy', 'except', 'select', 'intersect', 'having', 'union', 'limit'])
        sql = query['sql']

        query['names'] = full_query['names']
        #query['query_toks_no_value'] = ""
        query['col_table'] = full_query['col_table']
        query['col_set'] = full_query['col_set']
        query['table_names'] = full_query['table_names']
        query['keys'] = full_query['keys']
        query['columns_names_embedder'] = full_query['columns_names_embedder']
        query['columns_names_embedder_idxes'] = full_query['columns_names_embedder_idxes']
        query['question'] = query['utterance_toks']


        nest_query = {}

        nest_query['question'] = query['utterance_toks']
        nest_query['query'] = query['query']

        nest_query['utterance_arg_type'] = query['utterance_arg_type']
        nest_query['utterance_arg'] = query['utterance_arg']

        # nest_query['query_toks_no_value'] = query['query_toks_no_value']

        nest_query['names'] = full_query['names']
        nest_query['columns_names_embedder'] = full_query['columns_names_embedder']
        nest_query['columns_names_embedder_idxes'] = full_query['columns_names_embedder_idxes']
        nest_query['col_table'] = full_query['col_table']
        nest_query['col_set'] = full_query['col_set']
        nest_query['table_names'] = full_query['table_names']
        nest_query['keys'] = full_query['keys']


        if sql['intersect']:
            results = [Root1(0)]

            nest_query['sql'] = sql['intersect']
            results.extend(self.parser(query, history_utterance))
            results.extend(self.parser(nest_query, history_utterance))
            return results

        if sql['union']:
            results = [Root1(1)]
            nest_query['sql'] = sql['union']
            results.extend(self.parser(query, history_utterance))
            results.extend(self.parser(nest_query, history_utterance))
            return results

        if sql['except']:
            results = [Root1(2)]
            nest_query['sql'] = sql['except']
            results.extend(self.parser(query, history_utterance))
            results.extend(self.parser(nest_query, history_utterance))
            return results

        results = [Root1(3)]

        results.extend(self.parser(query, history_utterance))

        del query['names'],query['col_table'] ,query['col_set'] ,\
            query['table_names'],query['keys'] ,query['question'],\
            query['columns_names_embedder_idxes'],query['columns_names_embedder']

        return results

    def parser(self, query, history_utterance):
        stack = ["ROOT"]

        result = []
        # print('\n',query['sql'],'\n')
        while len(stack) > 0:
            state = stack.pop()

            step_result, step_state = self._parse_step(state, query, history_utterance)
            # print('before {} --> after {} , step res : {}'.format(state,step_state,step_result))
            result.extend(step_result)
            if step_state:
                stack.extend(step_state)
        return result

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, help='dataset', required=True)
    arg_parser.add_argument('--table_path', type=str, help='table dataset', required=True)
    arg_parser.add_argument('--output', type=str, help='output data', required=True)
    args = arg_parser.parse_args()
    # python3 sql2SemQL_test.py --data_path ../schema_linking_process/dev_no_value_schema_linking.json --table_path ../sparc/tables.json --output ../process_data/pre_dev.json
    parser = Parser()

    # loading dataSets
    datas, table = load_dataSets(args)
    processed_data = []
    #TODO dict_keys(['keys', 'interaction', 'final', 'col_set', 'table_names', 'names', 'col_table', 'database_id'])
    # print(datas[0].keys())

    global error_cnt,from_T
    error_cnt =0
    from_T = 0
    turn_cnt=0

    # f = open('rule_str.txt','w')
    group_by = 0
    group_by1 = 0
    for i, d in enumerate(datas):

        history_utterance = []
        for turn_idx,interaction in enumerate(datas[i]['interaction']):

            if len(datas[i]['interaction'][turn_idx]['sql']['select'][1]) > 6:
                continue

            # if 'GROUP BY' in datas[i]['interaction'][turn_idx]['query'] or 'group by' in datas[i]['interaction'][turn_idx]['query']:
            #     group_by+=1
            # if datas[i]['interaction'][turn_idx]['sql']['groupBy']!=[]:
            #     group_by1+=1
            # if ('GROUP BY' in datas[i]['interaction'][turn_idx]['query'] or 'group by' in datas[i]['interaction'][turn_idx]['query']) \
            #         and datas[i]['interaction'][turn_idx]['sql']['groupBy']==[]:
            #     print(datas[i]['interaction'][turn_idx]['utterance'])
            #     print(datas[i]['interaction'][turn_idx]['query'])
            #     print()

            # if datas[i]['interaction'][turn_idx]['query'] == 'SELECT count(*) FROM  INVESTORS AS T1 JOIN TRANSACTIONS AS T2 ON T1.investor_id  =  T2.investor_id WHERE T2.share_count  >  100':
            # if datas[i]['interaction'][turn_idx]['query'] == 'SELECT t1.name FROM people AS t1 JOIN candidate AS t2 ON t1.people_id  =  t2.people_id ORDER BY t2.support_rate DESC LIMIT 1':
            # if datas[i]['interaction'][turn_idx]['query'] == 'SELECT COUNT(*) FROM (SELECT cName FROM tryout WHERE pPos  =  \'goalie\' INTERSECT SELECT cName FROM  tryout WHERE pPos  =  \'mid\')':

            # if datas[i]['interaction'][turn_idx]['query'] == 'SELECT * FROM INVESTORS AS T1 JOIN TRANSACTIONS AS T2 ON T1.investor_id  =  T2.investor_id':

            history_utterance.append(datas[i]['interaction'][turn_idx])

            r = parser.full_parse(datas[i], datas[i]['interaction'][turn_idx], history_utterance)
            # print()
            # print(datas[i]['col_set'])
            # print(datas[i]['table_names'])
            # print(datas[i]['interaction'][turn_idx]['query'])
            # print(r)
            # exit(1)
            # print(datas[i]['interaction'][turn_idx]['query'])

            # if len(datas[i]['sql']['select'][1]) >=2 and datas[i]['sql']['where'] != [] and datas[i]['sql']['having']!=[]:
                # print(datas[i])
                # print('\n',datas[i]['query'])
                # print('\n\t\t',r)
                # exit(1)
            # print()
            #
            # print(datas[i]['interaction'][turn_idx]['query'])
            # print(datas[i]['table_names'])
            # print(datas[i]['names'])
            # print(datas[i]['col_table'])
            # print(r)
            # print()
            # f.write(" ".join([repr(x) for x in r])+'\n')
            # f.write("\t\t"+datas[i]['interaction'][turn_idx]['utterance']+'\n')
            # f.write("\t\t"+datas[i]['interaction'][turn_idx]['query']+'\n')
            '''for x in r:
                if x.parent!=None:
                    print(x,'->',str(x),'--->',x.parent,inv_action_map[x.parent])
                else:
                    print(x,'->',str(x),'--->',x.parent)
            #exit(1)
            print('\n')'''
            datas[i]['interaction'][turn_idx]['rule_label'] = " ".join([str(x) for x in r])
            turn_cnt +=1

            # processed_data.append(datas[i])
        # f.write('\n')

    # f.close()
    print(group_by,group_by1)
    print('From T : {}'.format(from_T))
    print('Finished %s datas and failed %s inferred table from *' % (turn_cnt,error_cnt))
    print('='*20)
    print('Limit match value : {} / {} [{} %]\n'
          'Filter Exact Match : {} / {} [{} %]'.format(limit_value_match,turn_cnt, round(limit_value_match/turn_cnt,3) * 100,
          filter_value_match, turn_cnt, round(filter_value_match/turn_cnt,3) * 100))
    print('Filter Part Match : {} / {} [{} %]'.format(filter_value_part_match,turn_cnt,round(filter_value_part_match/turn_cnt,3)*100))
    print('Filter Between Exact Match : {} / {} [{} %]'.format(filter_between_value_match, turn_cnt,
                                                      round(filter_between_value_match / turn_cnt, 3) * 100))
    print('Filter Column Exact Match : {} / {} [{} %]'.format(filter_column, turn_cnt,
                                                        round(filter_column / turn_cnt, 3) * 100))

    with open(args.output, 'w', encoding='utf8') as f:
        f.write(json.dumps(datas,indent=4))

