import os
import json
from src import utils
from .sparc_batch_feature import InteractionItem,UtteranceItem
import random,copy
import numpy as np
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

class SparcDataset():
    def __init__(self,args,mode='train'):
        if mode == 'train':
            self.sql_data, self.table_data, self.val_sql_data, self.val_table_data = utils.load_dataset(args.dataset, use_small=args.use_small)
        else:
            self.val_sql_data, self.val_table_data = utils.load_dataset(args.dataset, use_small=args.use_small)

        if args.print_info:
            if mode == 'train':
                print('Train sql_data size:{},train table numbers:{}\n'
                    'Dev sql_data size:{},dev table numbers:{}'.format(len(self.sql_data), len(self.table_data), len(self.val_sql_data),
                                                                     len(self.val_table_data)))
            else:
                print('Dev sql_data size:{},dev table numbers:{}'.format(len(self.val_sql_data),
                                                                         len(self.val_table_data)))


    def get_all_interactions(self,data : list,table_data : dict ,_type='train',sorted_by_length=False):
        # ints = [
        #     InteractionItem(d) for d in data
        # ]
        ints = []
        for idx,interaction in enumerate(data):
            #TODO interaction.keys() -> dict_keys(['columns_names_embedder_idxes', 'final', 'names', 'table_names', 'interaction', 'keys',
            #                   'database_id', 'col_set', 'col_table', 'columns_names_embedder'])
            # self.table_data[d['database_id']].keys() -> dict_keys(['table_names_original', 'primary_keys', 'db_id', 'foreign_keys',
            #                   'column_names', 'table_names', 'column_types', 'column_names_original'])
            # table_info = table_data[interaction['database_id']]

            # table exact match, column exact match, column partial match, count(*)
            schema_feature = np.zeros((len(interaction['columns_names_embedder']), 3), dtype=np.float32)

            for utterance in interaction['interaction']:
                #TODO utterance.keys() -> dict_keys(['sql', 'utterance_arg_type', 'utterance', 'utterance_arg', 'nltk_pos',
                #                           'query', 'utterance_toks', 'origin_utterance_toks', 'rule_label', 'query_toks_no_value'])
                # process_dict = self.process(interaction,utterance,table_info)

                # # table, column, agg, MORE, MOST, value
                question_feature = np.zeros((len(utterance['utterance_arg']), 6), dtype=np.float32)

                utter_schema_linking = copy.deepcopy(utterance['utterance_arg'])
                self.feature_schema_linking(utter_schema_linking, utterance['utterance_arg_type'],
                                            interaction['columns_names_embedder'],
                                            question_feature, schema_feature)

                utterance['utterance_arg_linking'] = utter_schema_linking
                # utterance['question_feature'] = question_feature
                utterance['schema_feature'] = schema_feature
                # self.schema_linking(process_dict['question_arg'], process_dict['question_arg_type'],
                #                process_dict['one_hot_type'], process_dict['col_set_type'], process_dict['col_set_iter'],interaction)
                # process_dict['col_set_iter'][0] = ['count', 'number', 'many']
                # print(process_dict.keys())
                # print(process_dict['question_arg'])
                # print(process_dict['col_set_iter'])

            ints.append(InteractionItem(interaction,_type))


        # t = sum([len(i.interaction) for i in ints])
        if sorted_by_length:
            return sorted(ints,key=lambda x:len(x.interaction),reverse=True)
        else:

            return ints

    def feature_schema_linking(self,question_arg,question_arg_type, schema, question_feature, schema_feature):
        # #column_exact, column partial,
        #     scheam_feature = np.zeros(len(tc),6)
        split_table_name_list = []
        split_column_name_list = []
        for s in schema:
            x = s.split(' ')
            if '.' in x:
                idx = x.index('.')
                split_table_name_list.append(x[:idx])
                split_column_name_list.append(x[idx + 1:])
            else:
                split_table_name_list.append([])
                split_column_name_list.append(x)
        assert len(split_table_name_list) == len(split_column_name_list)
        for count_q, t_q in enumerate(question_arg_type):
            t = t_q[0]
            if t == 'NONE':
                continue
            elif t == 'table':
                question_feature[count_q][0] = 1
                table_name = question_arg[count_q]
                self.get_table_match_list(table_name,split_table_name_list, schema_feature, 0 )
                self.set_column_match(table_name, split_column_name_list, schema_feature, 1, 'exact')
                self.set_column_match(table_name, split_column_name_list, schema_feature, 2, 'part')
                question_arg[count_q] = ['table'] + question_arg[count_q]
            elif t == 'col':
                question_feature[count_q][1] = 1
                try:
                    column_name = question_arg[count_q]
                    self.get_table_match_list(column_name, split_table_name_list, schema_feature, 0)
                    self.set_column_match(column_name, split_column_name_list, schema_feature, 1,'exact')
                    self.set_column_match(column_name, split_column_name_list, schema_feature, 2, 'part')

                    question_arg[count_q] = ['column'] + question_arg[count_q]
                except:
                    pass
            elif t == 'agg':
                question_feature[count_q][2] = 1
            elif t == 'MORE':
                question_feature[count_q][3] = 1
            elif t == 'MOST':
                question_feature[count_q][4] = 1
            elif t == 'value':
                question_feature[count_q][5] = 1
                question_arg[count_q] = ['value'] + question_arg[count_q]
            else:
                pass

    def get_table_match_list(self, table_name, split_table_name_list, schema_feature, idx):

        for i, t in enumerate(split_table_name_list):
            m = set(table_name) & set(t)
            if len(m) == 0:
                continue
            if len(m) == len(set(table_name)):  # Exact
                schema_feature[i][idx] += 3
            else:
                schema_feature[i][idx] += 1

    def set_column_match(self, table_name, split_table_name_list, schema_feature, idx, exact='exact'):

        for i, t in enumerate(split_table_name_list):
            m = set(table_name) & set(t)
            if len(m) == 0:
                continue
            if exact == 'exact' and len(m) == len(set(table_name)):  # Exact
                schema_feature[i][idx] += 3
            if exact == 'part':
                schema_feature[i][idx] += 1

    def simple_schema_linking(self,question_arg, question_arg_type):
        for count_q, t_q in enumerate(question_arg_type):
            t = t_q[0]
            if t == 'NONE':
                continue
            elif t == 'table':
                question_arg[count_q] = ['table'] + question_arg[count_q]
            elif t == 'col':
                question_arg[count_q] = ['column'] + question_arg[count_q]
            elif t=='value':
                question_arg[count_q] = ['value'] + question_arg[count_q]


    def schema_linking(self,question_arg, question_arg_type, one_hot_type, col_set_type, col_set_iter, sql):
        '''
        :param question_arg:      [['what'], ['is'], ['number'], ['of'], ['employee'], ['in'], ['each'], ['department'], ['?']]
        :param question_arg_type: [['NONE'], ['NONE'], ['NONE'], ['NONE'], ['NONE'], ['NONE'], ['NONE'], ['table'], ['NONE']]
        :param one_hot_type:       numpy -> (len(question_arg),6)  table,col,agg,MORE, MOST ，MORE
        :param col_set_type:       numpy -> (len(col_set_iter),4)
        :param col_set_iter:       [['*'], ['employee', 'id'], ['name'], ['position'], ['ssn'], ['departmentid'], ['head'], ... ]
        :param sql:
        :return:
        '''
        for count_q,t_q in enumerate(question_arg_type):
            t = t_q[0]
            if t == 'NONE':
                continue
            elif t == 'table':
                one_hot_type[count_q][0] = 1
                question_arg[count_q] = ['table'] + question_arg[count_q]
            elif t == 'col':
                one_hot_type[count_q][1] = 1
                try:
                    #TODO ??????? 为什么是5
                    col_set_type[col_set_iter.index(question_arg[count_q])][1] = 5
                    question_arg[count_q] = ['column'] + question_arg[count_q]
                except:
                    print(col_set_iter,question_arg[count_q])
                    raise RuntimeError('not in col set')
            elif  t == 'agg':
                one_hot_type[count_q][2] = 1
            elif t == 'MORE':
                one_hot_type[count_q][3] = 1
            elif t == 'MOST':
                one_hot_type[count_q][4] = 1
            elif t=='value':
                one_hot_type[count_q][5] = 1
                question_arg[count_q] = ['value'] + question_arg[count_q]
            else:
                if len(t_q) == 1:
                    for col_probase in t_q:
                        if col_probase == 'asd':
                            continue
                        try:
                            col_set_type[sql['col_set'].index(col_probase)][2] = 5
                            question_arg[count_q] = ['value'] + question_arg[count_q]
                        except:
                            print(sql['col_set'], col_probase)
                            raise RuntimeError('not in col')
                        one_hot_type[count_q][5] = 1
                else:
                    for col_probase in t_q:
                        if col_probase == 'asd':
                            continue
                        col_set_type[sql['col_set'].index(col_probase)][3] += 1



    def process(self,interaction,utterance,table):
        process_dict = {}
        origin_sql = utterance['utterance_toks']

        table_names = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(' ')] for x in table['table_names']]

        utterance['pre_sql'] = copy.deepcopy(utterance)

        tab_cols = [col[1] for col in table['column_names']]
        tab_ids = [col[0] for col in table['column_names']]


        col_set_iter = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(' ')] for x in interaction['col_set']]
        col_iter = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(" ")] for x in tab_cols]

        q_iter_small = [wordnet_lemmatizer.lemmatize(x).lower() for x in origin_sql]

        question_arg = copy.deepcopy(utterance['utterance_arg'])
        question_arg_type = utterance['utterance_arg_type']
        # TODO 6-> NONE不算 ,  table,col,agg,value, MOST ，MORE
        one_hot_type = np.zeros((len(question_arg_type), 6))
        # TODO 4-> exact match , partial match , value exact match , value partial match
        col_set_type = np.zeros((len(col_set_iter), 4))

        process_dict['col_set_iter'] = col_set_iter
        process_dict['q_iter_small'] = q_iter_small
        process_dict['col_set_type'] = col_set_type
        process_dict['question_arg'] = question_arg
        process_dict['question_arg_type'] = question_arg_type
        process_dict['one_hot_type'] = one_hot_type
        process_dict['tab_cols'] = tab_cols
        process_dict['tab_ids'] = tab_ids
        process_dict['col_iter'] = col_iter
        process_dict['table_names'] = table_names

        return process_dict



    def get_all_utterances(self,data : list):
        items = []
        for interaction in data:
            for i,utterance in enumerate(interaction['interaction']):
                items.append(UtteranceItem(utterance))

        return items

    def get_interaction_batches(self,batch_size,randomize=True,sample_num=None):
        items = self.get_all_interactions(self.sql_data,self.table_data)
        if randomize:
            random.shuffle(items)

        if sample_num:
            sample_items = items[:sample_num]

        batchs = []
        current_batch_items = []
        for item in items:
            if len(current_batch_items) >= batch_size:
                batchs.append(current_batch_items)
                current_batch_items = []
            current_batch_items.append(item)
        batchs.append(current_batch_items)

        assert sum([len(batch) for batch in batchs]) == len(items)

        return batchs,sample_items




    def get_utterance_batches(self,batch_size):
        items = self.get_all_utterances(self.sql_data)


