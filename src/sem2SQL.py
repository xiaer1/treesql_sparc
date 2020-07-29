# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# -*- coding: utf-8 -*-
"""
# @Time    : 2019/12/06
# @Author  : xiaxia.wang
# @File    : sem2SQL.py
# @Software: PyCharm
"""

import argparse,json
import traceback
import regex
import sqlparse
import sqlite3
import numpy as np
import logger
import sys,os
sys.path.insert(0,os.getcwd())
from src.rule.graph import Graph
from src.rule.semQL import Sup, Sel, Order, Root, Filter, A, N, C, T, Root1,Turn,ValueS,Distance
from src.rule.sem_utils import alter_inter, alter_not_in, alter_column0, load_dataSets
from src.parse_sql_py import parse_sql
DB_DIR =  os.path.join('sparc','database')


log_dir = 'log_tmp'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
record = logger.Logger(os.path.join(log_dir, os.path.basename(__file__).split('.')[0] + '.log'), 'w')

def split_logical_form(lf):
    indexs = [i+1 for i, letter in enumerate(lf) if letter == ')']
    indexs.insert(0, 0)
    components = list()
    for i in range(1, len(indexs)):
        components.append(lf[indexs[i-1]:indexs[i]].strip())
    return components


def pop_front(array):
    if len(array) == 0:
        return 'None'
    return array.pop(0)


def is_end(components, transformed_sql, is_root_processed):
    end = False
    c = pop_front(components)
    c_instance = eval(c)

    if isinstance(c_instance, Root) and is_root_processed:
        # intersect, union, except
        end = True
    elif isinstance(c_instance, Filter):
        if 'where' not in transformed_sql:
            end = True
        else:
            num_conjunction = 0
            for f in transformed_sql['where']:
                if isinstance(f, str) and (f == 'and' or f == 'or'):
                    num_conjunction += 1
            current_filters = len(transformed_sql['where'])
            valid_filters = current_filters - num_conjunction
            if valid_filters >= num_conjunction + 1:
                end = True
    elif isinstance(c_instance, Order):
        if 'order' not in transformed_sql:
            end = True
        elif len(transformed_sql['order']) == 0:
            end = False
        else:
            end = True
    elif isinstance(c_instance, Sup):
        if 'sup' not in transformed_sql:
            end = True
        elif len(transformed_sql['sup']) == 0:
            end = False
        else:
            end = True
    components.insert(0, c)
    return end

def string_distance(str1, str2):
    """
    计算两个字符串之间的编辑距离
    @author: 仰起脸笑的像满月
    @date: 2019/05/15
    :param str1:
    :param str2:
    :return:
    """
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


def get_max_match_score(p_res, value):
    res = [x[0].lower() for x in p_res]
    v = value.lower()
    score_dict = dict()
    for x, y in zip(res, p_res):
        score_dict[y[0]] = string_distance(v, x)
    score_dict = sorted(score_dict.items(), key=lambda x: x[1])
    if score_dict[0][1] <= 2:
        record.put('Origin Value : {} ---> search Value : {}\n'.format(value, score_dict[0][0]))
        value = score_dict[0][0]
    return value

def search_db_content(truth_value, db_name, pred_str):
    db = os.path.join(DB_DIR, db_name, db_name + ".sqlite")
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    try:

        cursor.execute(pred_str)
        p_res = cursor.fetchall()

        truth_value = get_max_match_score(p_res, truth_value)
    except:

        pass

    return truth_value

def _transform(components, transformed_sql, col_set, table_names, schema, utter_list):
    processed_root = False
    current_table = schema
    while len(components) > 0:

        if is_end(components, transformed_sql, processed_root):
            break
        c = pop_front(components)
        c_instance = eval(c)
        if isinstance(c_instance, Root):
            processed_root = True
            transformed_sql['select'] = list()
            if c_instance.id_c == 0:
                transformed_sql['where'] = list()
                transformed_sql['sup'] = list()
            elif c_instance.id_c == 1:
                transformed_sql['where'] = list()
                transformed_sql['order'] = list()
            elif c_instance.id_c == 2:
                transformed_sql['sup'] = list()
            elif c_instance.id_c == 3:
                transformed_sql['where'] = list()
            elif c_instance.id_c == 4:
                transformed_sql['order'] = list()
        elif isinstance(c_instance, Sel):
            continue
        elif isinstance(c_instance, N):
            for i in range(c_instance.id_c + 1):
                agg = eval(pop_front(components))
                column = eval(pop_front(components))
                _table = pop_front(components)
                table = eval(_table)
                if not isinstance(table, T):
                    table = None
                    components.insert(0, _table)
                assert isinstance(agg, A) and isinstance(column, C)

                if table.id_c == -1:
                    table = None

                transformed_sql['select'].append((
                    agg.production.split()[1],
                    replace_col_with_original_col(col_set[column.id_c], table_names[table.id_c], current_table) if table is not None else col_set[column.id_c],
                    table_names[table.id_c] if table is not None else table
                ))

        elif isinstance(c_instance, Sup):
            transformed_sql['sup'].append(c_instance.production.split()[1])
            agg = eval(pop_front(components))
            column = eval(pop_front(components))
            _table = pop_front(components)
            table = eval(_table)
            if not isinstance(table, T):
                table = None
                components.insert(0, _table)
            assert isinstance(agg, A) and isinstance(column, C)

            if table.id_c == -1:
                table = None

            transformed_sql['sup'].append(agg.production.split()[1])

            fix_col_id = replace_col_with_original_col(col_set[column.id_c], table_names[table.id_c],current_table) if table is not None else col_set[column.id_c]
            transformed_sql['sup'].append(fix_col_id)
            transformed_sql['sup'].append(table_names[table.id_c] if table is not None else table)

            _turn = eval(pop_front(components))
            _valueS = eval(pop_front(components))
            _distance = eval(pop_front(components))
            if _distance.id_c != 0:
                print('Limit error')

            single_utter = utter_list[_turn.id_c]
            value_list = single_utter[_valueS.id_c : _valueS.id_c  + 1]

            truth_value = ' '.join([' '.join(x) for x in value_list])
            if not truth_value.isdigit():
                truth_value = '1'
            transformed_sql['sup'].append(truth_value)


        elif isinstance(c_instance, Order):
            transformed_sql['order'].append(c_instance.production.split()[1])
            agg = eval(pop_front(components))
            column = eval(pop_front(components))
            _table = pop_front(components)
            table = eval(_table)
            if not isinstance(table, T):
                table = None
                components.insert(0, _table)
            assert isinstance(agg, A) and isinstance(column, C)

            if table.id_c == -1:
                table = None

            transformed_sql['order'].append(agg.production.split()[1])
            transformed_sql['order'].append(replace_col_with_original_col(col_set[column.id_c], table_names[table.id_c], current_table) if table is not None else col_set[column.id_c])
            transformed_sql['order'].append(table_names[table.id_c] if table is not None else table)

        elif isinstance(c_instance, Filter):
            op = c_instance.production.split()[1]
            if op == 'and' or op == 'or':
                transformed_sql['where'].append(op)
            else:
                # No Supquery
                agg = eval(pop_front(components))
                column = eval(pop_front(components))
                _table = pop_front(components)
                table = eval(_table)
                if not isinstance(table, T):
                    table = None
                    components.insert(0, _table)
                assert isinstance(agg, A) and isinstance(column, C)

                if table.id_c == -1:
                    table = None
                if c_instance.id_c>=2 and c_instance.id_c<=10:
                    #TODO  op, agg, col, tab, value = f
                    _turn = eval(pop_front(components))
                    _valueS = eval(pop_front(components))
                    _distance = eval(pop_front(components))

                    single_utter = utter_list[_turn.id_c]
                    value_list = single_utter[_valueS.id_c: _valueS.id_c + _distance.id_c + 1]

                    truth_value = ' '.join([' '.join(x) for x in value_list])

                    if not truth_value.isdigit():
                        db_column_name = replace_col_with_original_col(col_set[column.id_c], table_names[table.id_c],current_table) \
                                             if table is not None else col_set[column.id_c],
                        db_tab_name = table_names[table.id_c] if table is not None else table,
                        pred_str = 'SELECT {} FROM {}'.format(db_column_name[0], db_tab_name[0])
                        db_name = current_table['db_id']
                        truth_value = search_db_content(truth_value, db_name, pred_str)

                    if c_instance.id_c == 9 or c_instance.id_c == 10: #like
                        truth_value = '%' + truth_value + '%'
                    if not truth_value.isdigit():
                        truth_value =  '\"' + truth_value + '\"'

                    truth_value1 = None
                    if c_instance.id_c == 8: #between
                        _turn1 = eval(pop_front(components))
                        _valueS1 = eval(pop_front(components))
                        _distance1 = eval(pop_front(components))
                        single_utter1 = utter_list[_turn1.id_c]
                        value_list1 = single_utter1[_valueS1.id_c: _valueS1.id_c + _distance1.id_c + 1]

                        truth_value1 = ' '.join([' '.join(x) for x in value_list1])
                        if c_instance.id_c == 9 or c_instance.id_c == 10:  # like
                            truth_value1 = '%' + truth_value1 + '%'
                        if not truth_value1.isdigit():
                            truth_value1 = '\"' + truth_value1 + '\"'

                        if not truth_value1.isdigit():
                            db_column_name = replace_col_with_original_col(col_set[column.id_c],
                                                                           table_names[table.id_c], current_table) \
                                                 if table is not None else col_set[column.id_c],
                            db_tab_name = table_names[table.id_c] if table is not None else table,
                            pred_str = 'SELECT {} FROM {}'.format(db_column_name[0], db_tab_name[0])
                            db_name = current_table['db_id']
                            truth_value1 = search_db_content(truth_value1, db_name, pred_str)


                    transformed_sql['where'].append((
                        op,
                        agg.production.split()[1],
                        replace_col_with_original_col(col_set[column.id_c], table_names[table.id_c], current_table) if table is not None else col_set[column.id_c],
                        table_names[table.id_c] if table is not None else table,
                        truth_value,
                        truth_value1,
                        None
                    ))
                else: #11 - 19
                    # Subquery

                    new_dict = dict()
                    new_dict['sql'] = transformed_sql['sql']
                    transformed_sql['where'].append((
                        op,
                        agg.production.split()[1],
                        replace_col_with_original_col(col_set[column.id_c], table_names[table.id_c], current_table) if table is not None else col_set[column.id_c],
                        table_names[table.id_c] if table is not None else table,
                        None,
                        None,
                        _transform(components, new_dict, col_set, table_names, schema, utter_list)
                    ))

    return transformed_sql


def transform(interaction, utterance, utter_list, schema, origin=None):
    '''

    :param interaction: dict_keys(['col_set', 'columns_names_embedder_idxes', 'table_names', 'col_table', 'names',
                    'columns_names_embedder', 'database_id', 'interaction', 'final', 'keys'])
    :param utterance: dict_keys(['origin_utterance_toks', 'rule_label', 'utterance_arg_linking', 'pred_lf_replace',
            'utterance_arg', 'utterance_toks', 'sql', 'query', 'query_toks_no_value', 'utterance',
            'rule_count', 'pred_sketch', 'pred_lf', 'utterance_arg_type', 'target_lf', 'nltk_pos'])
    :param schema:
    :param origin:
    :return:
    '''
    preprocess_schema(schema)

    # if origin is None:
    #     lf = utterance['pred_lf_replace']
    # else:
    #     lf = origin

    lf = utterance['pred_lf']

    col_set = interaction['col_set']
    table_names = interaction['table_names']
    current_table = schema

    current_table['schema_content_clean'] = [x[1] for x in current_table['column_names']]
    current_table['schema_content'] = [x[1] for x in current_table['column_names_original']]

    components = split_logical_form(lf)

    transformed_sql = dict()
    transformed_sql['sql'] = utterance
    c = pop_front(components)
    c_instance = eval(c)
    if not isinstance(c_instance, Root1):
        print('Not root1')
        return ['']

    if c_instance.id_c == 0:
        transformed_sql['intersect'] = dict()
        transformed_sql['intersect']['sql'] = utterance

        _transform(components, transformed_sql, col_set, table_names, schema, utter_list)
        _transform(components, transformed_sql['intersect'], col_set, table_names, schema, utter_list)
    elif c_instance.id_c == 1:
        transformed_sql['union'] = dict()
        transformed_sql['union']['sql'] = utterance
        _transform(components, transformed_sql, col_set, table_names, schema, utter_list)
        _transform(components, transformed_sql['union'], col_set, table_names, schema, utter_list)
    elif c_instance.id_c == 2:
        transformed_sql['except'] = dict()
        transformed_sql['except']['sql'] = utterance
        _transform(components, transformed_sql, col_set, table_names, schema,utter_list)
        _transform(components, transformed_sql['except'], col_set, table_names, schema,utter_list)
    else:
        _transform(components, transformed_sql, col_set, table_names, schema,utter_list)

    parse_result = to_str(transformed_sql, 1, schema)

    parse_result = parse_result.replace('\t', '')

    return [parse_result]

def col_to_str(agg, col, tab, table_names, N=1):
    _col = col.replace(' ', '_')
    if agg == 'none':
        if tab not in table_names:
            table_names[tab] = 'T' + str(len(table_names) + N)
        table_alias = table_names[tab]
        if col == '*':
            return '*'
        return '%s.%s' % (table_alias, _col)
    else:
        if col == '*':
            if tab is not None and tab not in table_names:
                table_names[tab] = 'T' + str(len(table_names) + N)
            return '%s(%s)' % (agg, _col)
        else:
            if tab not in table_names:
                table_names[tab] = 'T' + str(len(table_names) + N)
            table_alias = table_names[tab]
            return '%s(%s.%s)' % (agg, table_alias, _col)


def infer_from_clause(table_names, schema, columns):
    tables = list(table_names.keys())
    # print(table_names)
    start_table = None
    end_table = None
    join_clause = list()
    if len(tables) == 1:
        join_clause.append((tables[0], table_names[tables[0]]))
    elif len(tables) == 2:
        use_graph = True
        # print(schema['graph'].vertices)
        for t in tables:
            if t not in schema['graph'].vertices:
                use_graph = False
                break
        if use_graph:
            start_table = tables[0]
            end_table = tables[1]
            _tables = list(schema['graph'].dijkstra(tables[0], tables[1]))
            if _tables == []:
                print('null tables')
                for t in tables:
                    join_clause.append((t, table_names[t],))
            # print('Two tables: ', _tables)
            max_key = 1
            for t, k in table_names.items():
                _k = int(k[1:])
                if _k > max_key:
                    max_key = _k
            for t in _tables:
                if t not in table_names:
                    table_names[t] = 'T' + str(max_key + 1)
                    max_key += 1
                join_clause.append((t, table_names[t],))
        else:
            join_clause = list()
            for t in tables:
                join_clause.append((t, table_names[t],))
    else:
        # > 2
        # print('More than 2 table')
        for t in tables:
            join_clause.append((t, table_names[t],))

    if len(join_clause) >= 3:
        star_table = None
        for agg, col, tab in columns:
            if col == '*':
                star_table = tab
                break
        if star_table is not None:
            star_table_count = 0
            for agg, col, tab in columns:
                if tab == star_table and col != '*':
                    star_table_count += 1
            if star_table_count == 0 and ((end_table is None or end_table == star_table) or (start_table is None or start_table == star_table)):
                # Remove the table the rest tables still can join without star_table
                new_join_clause = list()
                for t in join_clause:
                    if t[0] != star_table:
                        new_join_clause.append(t)
                join_clause = new_join_clause

    join_clause = ' JOIN '.join(['%s AS %s' % (jc[0], jc[1]) for jc in join_clause])
    return 'FROM ' + join_clause

def replace_col_with_original_col(query, col, current_table):

    if query == '*':
        return query

    cur_table = col
    cur_col = query
    single_final_col = cur_col

    for col_ind, col_name in enumerate(current_table['schema_content_clean']):
        if col_name == cur_col:
            #assert cur_table in current_table['table_names']
            if current_table['table_names'][current_table['col_table'][col_ind]] == cur_table:
                single_final_col = current_table['column_names_original'][col_ind][1]
                break

    #assert single_final_col
    # if query != single_final_col:
    #     print(query, single_final_col)
    return single_final_col


def build_graph(schema):
    relations = list()
    foreign_keys = schema['foreign_keys']
    for (fkey, pkey) in foreign_keys:
        fkey_table = schema['table_names_original'][schema['column_names'][fkey][0]]
        pkey_table = schema['table_names_original'][schema['column_names'][pkey][0]]
        relations.append((fkey_table, pkey_table))
        relations.append((pkey_table, fkey_table))
    return Graph(relations)


def preprocess_schema(schema):
    tmp_col = []
    for cc in [x[1] for x in schema['column_names']]:
        if cc not in tmp_col:
            tmp_col.append(cc)
    schema['col_set'] = tmp_col
    # print table
    schema['schema_content'] = [col[1] for col in schema['column_names']]
    schema['col_table'] = [col[0] for col in schema['column_names']]

    graph = build_graph(schema)

    schema['graph'] = graph




def to_str(sql_json, N_T, schema, pre_table_names=None):
    all_columns = list()
    select_clause = list()
    table_names = dict()
    current_table = schema
    for (agg, col, tab) in sql_json['select']:
        all_columns.append((agg, col, tab))
        select_clause.append(col_to_str(agg, col, tab, table_names, N_T))
    select_clause_str = 'SELECT ' + ', '.join(select_clause).strip()

    sup_clause = ''
    order_clause = ''
    direction_map = {"des": 'DESC', 'asc': 'ASC'}

    if 'sup' in sql_json:
        (direction, agg, col, tab, val) = sql_json['sup']

        all_columns.append((agg, col, tab))
        subject = col_to_str(agg, col, tab, table_names, N_T)
        sup_clause = ('ORDER BY %s %s LIMIT %s' % (subject, direction_map[direction], val)).strip()

    elif 'order' in sql_json:
        (direction, agg, col, tab,) = sql_json['order']
        all_columns.append((agg, col, tab))
        subject = col_to_str(agg, col, tab, table_names, N_T)
        order_clause = ('ORDER BY %s %s' % (subject, direction_map[direction])).strip()

    has_group_by = False
    where_clause = ''
    have_clause = ''
    if 'where' in sql_json:
        conjunctions = list()
        filters = list()

        for f in sql_json['where']:

            if isinstance(f, str):
                conjunctions.append(f)
            else:
                op, agg, col, tab, val1, val2 , value = f
                if value:
                    value['sql'] = sql_json['sql']
                all_columns.append((agg, col, tab))

                subject = col_to_str(agg, col, tab, table_names, N_T)

                if value is None:
                    where_value = '%s' % (val1)
                    if op == 'between':
                        where_value = '%s AND %s' % (val1, val2)
                    filters.append('%s %s %s' % (subject, op, where_value))
                else:
                    if op == 'in' and len(value['select']) == 1 and value['select'][0][0] == 'none' \
                            and 'where' not in value and 'order' not in value and 'sup' not in value:
                            # and value['select'][0][2] not in table_names:
                        if value['select'][0][2] not in table_names:
                            table_names[value['select'][0][2]] = 'T' + str(len(table_names) + N_T)
                        filters.append(None)

                    else:
                        filters.append('%s %s %s' % (subject, op, '(' + to_str(value, len(table_names) + 1, schema) + ')'))

                if len(conjunctions):
                    filters.append(conjunctions.pop())

        aggs = ['count(', 'avg(', 'min(', 'max(', 'sum(']
        having_filters = list()
        idx = 0
        while idx < len(filters):
            _filter = filters[idx]
            if _filter is None:
                idx += 1
                continue
            for agg in aggs:
                if _filter.startswith(agg):
                    having_filters.append(_filter)
                    filters.pop(idx)
                    # print(filters)
                    if 0 < idx and (filters[idx - 1] in ['and', 'or']):
                        filters.pop(idx - 1)
                        # print(filters)
                    break
            else:
                idx += 1
        if len(having_filters) > 0:
            have_clause = 'HAVING ' + ' '.join(having_filters).strip()
        if len(filters) > 0:

            filters = [_f for _f in filters if _f is not None]
            conjun_num = 0
            filter_num = 0
            for _f in filters:
                if _f in ['or', 'and']:
                    conjun_num += 1
                else:
                    filter_num += 1
            if conjun_num > 0 and filter_num != (conjun_num + 1):
                # assert 'and' in filters
                idx = 0
                while idx < len(filters):
                    if filters[idx] == 'and':
                        if idx - 1 == 0:
                            filters.pop(idx)
                            break
                        if filters[idx - 1] in ['and', 'or']:
                            filters.pop(idx)
                            break
                        if idx + 1 >= len(filters) - 1:
                            filters.pop(idx)
                            break
                        if filters[idx + 1] in ['and', 'or']:
                            filters.pop(idx)
                            break
                    idx += 1
            if len(filters) > 0:
                where_clause = 'WHERE ' + ' '.join(filters).strip()
                where_clause = where_clause.replace('not_in', 'NOT IN')
            else:
                where_clause = ''

        if len(having_filters) > 0:
            has_group_by = True

    for agg in ['count(', 'avg(', 'min(', 'max(', 'sum(']:
        if (len(sql_json['select']) > 1 and agg in select_clause_str)\
                or agg in sup_clause or agg in order_clause:
            has_group_by = True
            break

    group_by_clause = ''
    if has_group_by:

        if len(table_names) == 1:
            # check none agg
            is_agg_flag = False
            for (agg, col, tab) in sql_json['select']:

                if agg == 'none':
                    group_by_clause = 'GROUP BY ' + col_to_str(agg, col, tab, table_names, N_T)
                else:
                    is_agg_flag = True

            if is_agg_flag is False and len(group_by_clause) > 5:
                group_by_clause = "GROUP BY"
                for (agg, col, tab) in sql_json['select']:
                    group_by_clause = group_by_clause + ' ' + col_to_str(agg, col, tab, table_names, N_T)

            if len(group_by_clause) < 5:
                if 'count(*)' in select_clause_str:
                    current_table = schema
                    for primary in current_table['primary_keys']:
                        if current_table['table_names'][current_table['col_table'][primary]] in table_names :
                            group_by_clause = 'GROUP BY ' + col_to_str('none', current_table['schema_content'][primary],
                                                                       current_table['table_names'][
                                                                           current_table['col_table'][primary]],
                                                                       table_names, N_T)
        else:
            # if only one select

            if len(sql_json['select']) == 1:
                agg, col, tab = sql_json['select'][0]
                non_lists = [tab]
                fix_flag = False
                # add tab from other part
                for key, value in table_names.items():
                    if key not in non_lists:
                        non_lists.append(key)

                a = non_lists[0]
                b = None

                for non in non_lists:
                    if a != non:
                        b = non

                if b:

                    for pair in current_table['foreign_keys']:
                        t1 = current_table['table_names'][current_table['col_table'][pair[0]]]
                        t2 = current_table['table_names'][current_table['col_table'][pair[1]]]

                        if t1 in [a, b] and t2 in [a, b]:
                            if pre_table_names and t1 not in pre_table_names:
                                # assert t2 in pre_table_names
                                t1 = t2
                            group_by_clause = 'GROUP BY ' + col_to_str('none',
                                                                       current_table['schema_content'][pair[0]],
                                                                       t1,
                                                                       table_names, N_T)
                            fix_flag = True
                            break

                if fix_flag is False:
                    agg, col, tab = sql_json['select'][0]
                    group_by_clause = 'GROUP BY ' + col_to_str(agg, col, tab, table_names, N_T)

            else:
                # check if there are only one non agg
                non_agg, non_agg_count = None, 0
                non_lists = []
                for (agg, col, tab) in sql_json['select']:
                    if agg == 'none':
                        non_agg = (agg, col, tab)
                        non_lists.append(tab)
                        non_agg_count += 1

                non_lists = list(set(non_lists))

                if non_agg_count == 1:
                    group_by_clause = 'GROUP BY ' + col_to_str(non_agg[0], non_agg[1], non_agg[2], table_names, N_T)
                elif non_agg:
                    find_flag = False
                    fix_flag = False
                    find_primary = None
                    if len(non_lists) <= 1:
                        for key, value in table_names.items():
                            if key not in non_lists:
                                non_lists.append(key)
                    if len(non_lists) > 1:
                        a = non_lists[0]
                        b = None
                        for non in non_lists:
                            if a != non:
                                b = non

                        if b:

                            for pair in current_table['foreign_keys']:

                                t1 = current_table['table_names'][current_table['col_table'][pair[0]]]
                                t2 = current_table['table_names'][current_table['col_table'][pair[1]]]
                                # print(t1,current_table['schema_content'][pair[0]], '--->',
                                #       t2,current_table['schema_content'][pair[1]])

                                if t1 in [a, b] and t2 in [a, b]:
                                    if pre_table_names and t1 not in pre_table_names:
                                        #assert  t2 in pre_table_names
                                        t1 = t2
                                    group_by_clause = 'GROUP BY ' + col_to_str('none',
                                                                               current_table['schema_content'][pair[0]],
                                                                               t1,
                                                                               table_names, N_T)
                                    fix_flag = True
                                    break

                    tab = non_agg[2]
                    #assert tab in current_table['table_names']

                    for primary in current_table['primary_keys']:
                        if current_table['table_names'][current_table['col_table'][primary]] == tab:
                            find_flag = True
                            find_primary = (current_table['schema_content'][primary], tab)
                    if fix_flag is False:
                        if find_flag is False:
                            # rely on count *
                            foreign = []
                            for pair in current_table['foreign_keys']:
                                if current_table['table_names'][current_table['col_table'][pair[0]]] == tab:
                                    foreign.append(pair[1])
                                if current_table['table_names'][current_table['col_table'][pair[1]]] == tab:
                                    foreign.append(pair[0])

                            for pair in foreign:
                                if current_table['table_names'][current_table['col_table'][pair]] in table_names:
                                    group_by_clause = 'GROUP BY ' + col_to_str('none', current_table['schema_content'][pair],
                                                                                   current_table['table_names'][current_table['col_table'][pair]],
                                                                                   table_names, N_T)
                                    find_flag = True
                                    break

                            if find_flag is False:

                                for (agg, col, tab) in sql_json['select']:
                                    if 'id' in col.lower():
                                        group_by_clause = 'GROUP BY ' + col_to_str(agg, col, tab, table_names, N_T)
                                        break

                                if len(group_by_clause) > 5:
                                    pass
                                else:
                                    pass
                                    # raise RuntimeError('fail to convert')
                        else:
                            group_by_clause = 'GROUP BY ' + col_to_str('none', find_primary[0],
                                                                       find_primary[1],
                                                                       table_names, N_T)

    intersect_clause = ''
    if 'intersect' in sql_json:
        sql_json['intersect']['sql'] = sql_json['sql']
        intersect_clause = 'INTERSECT ' + to_str(sql_json['intersect'], len(table_names) + 1, schema, table_names)
    union_clause = ''
    if 'union' in sql_json:
        sql_json['union']['sql'] = sql_json['sql']
        union_clause = 'UNION ' + to_str(sql_json['union'], len(table_names) + 1, schema, table_names)
    except_clause = ''
    if 'except' in sql_json:
        sql_json['except']['sql'] = sql_json['sql']
        except_clause = 'EXCEPT ' + to_str(sql_json['except'], len(table_names) + 1, schema, table_names)


    table_names_replace = {}
    for a, b in zip(current_table['table_names_original'], current_table['table_names']):
        table_names_replace[b] = a
    new_table_names = {}
    #print(table_names_replace)
    for key, value in table_names.items():
        if key is None:
            continue
        new_table_names[table_names_replace[key]] = value
    from_clause = infer_from_clause(new_table_names, schema, all_columns).strip()

    sql = ' '.join([select_clause_str, from_clause, where_clause, group_by_clause, have_clause, sup_clause, order_clause,
                    intersect_clause, union_clause, except_clause])

    return sql

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

def parse_sql_to_std(query, column_names, schema_tokens, schemas):
    turn_sql = []
    skip = False
    for query_tok in query:
        if query_tok != '.' and '.' in query_tok:
            turn_sql += query_tok.replace('.', ' . ').split()
            skip = True
        else:
            turn_sql.append(query_tok)

    turn_sql = ' '.join(turn_sql)

    turn_sql_parse = parse_sql(turn_sql, column_names, schema_tokens, schemas)


    return turn_sql_parse

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, help='dataset path', required=True)
    arg_parser.add_argument('--input_path', type=str, help='predicted logical form', required=True)
    args = arg_parser.parse_args()

    # loading dataSets
    #TODO datas : list , schemas : dict
    #  schemas.keys() -> dict_keys(['entertainment_awards', 'performance_attendance', 'local_govt_mdm', 'manufacturer', 'store_product', 'debate', 'poker_player', 'phone_1', 'school_bus', 'cre_Theme_park', 'manufactory_1', 'chinook_1', 'body_builder',
    # datas[0].keys() -> dict_keys(['question_toks', 'names', 'question', 'question_arg_type', 'keys', 'table_names', 'query_toks', 'sql', 'query_toks_no_value', 'sketch_result', 'rule_label', 'question_arg', 'db_id', 'query', 'col_table', 'nltk_pos', 'col_set', 'model_result', 'origin_question_toks'])
    # new -> dict_keys(['final', 'columns_names_embedder_idxes', 'col_table', 'database_id', 'interaction', 'table_names', 'keys', 'names', 'col_set', 'columns_names_embedder'])
    datas, schemas = load_dataSets(args)

    #alter_not_in(datas, schemas=schemas)
    #alter_inter(datas)
    # alter_column0(datas)
    # python3 src/sem2SQL2.py --data_path data/sparc/ --input_path predict_lf.json
    index = range(len(datas))
    count = 0
    exception_count = 0

    # error = open('error2.txt','w')
    # error_not_same = open('error_not_same.txt','w')
    prediction_list = []
    predict_f = open("prediction.json", "w")

    for i,interaction in enumerate(datas):

        for utter_idx,utter in enumerate(interaction['interaction']):
            utter_dict = {}
            utter_dict['identifier'] = datas[i]['database_id']+'/'+str(i)
            utter_dict['database_id'] = datas[i]['database_id']
            utter_dict['interaction_id'] = str(i)
            utter_dict['index_in_interaction'] = utter_idx
            utter_dict['input_seq'] = utter['origin_utterance_toks']

            schema_tokens, column_names = get_schema_tokens(schemas[datas[i]['database_id']])

            result = transform(datas[i], utter , schemas[datas[i]['database_id']])

            if result[0]!='':
                t = result[0].replace('.', ' . ').replace(',', ' , ').replace('(', ' ( ').replace(')', ' ) ').replace('>=', '> =').replace('<=', '< =')
                m = [x.lower() for x in t.split(' ') if x!='']

                pred_turn_sql_parse = parse_sql_to_std(m, column_names, schema_tokens,schemas[datas[i]['database_id']])

                if 't1.' in pred_turn_sql_parse or 't2.' in pred_turn_sql_parse or 't3.' in pred_turn_sql_parse or\
                        't4.' in pred_turn_sql_parse or 't5.' in pred_turn_sql_parse or 't6.' in pred_turn_sql_parse:
                    print('paser error with t_x')
                    utter_dict['flat_prediction'] = ['select']
                else:
                    utter_dict['flat_prediction'] = pred_turn_sql_parse.split(' ')
            else:
                utter_dict['flat_prediction'] = ['select']

            predict_f.write(json.dumps(utter_dict) +'\n')

    predict_f.close()
