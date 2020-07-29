from nltk.stem import WordNetLemmatizer
import src.rule.semQL as define_rule
from src.rule.semQL import Sup, Sel, Order, Root, Filter, A, N, Root1,TC, Turn,ValueS,Distance
import copy
import numpy as np
wordnet_lemmatizer = WordNetLemmatizer()

class InteractionItem():
    def __init__(self,interaction : dict,_type='train'):
        '''
        :param interaction:  dict_keys(['names', 'database_id', 'columns_names_embedder_idxes',
        'keys', 'col_set', 'col_table', 'columns_names_embedder', 'final', 'table_names', 'interaction'])

        '''
        if _type == 'test':
            self.full_interaction = interaction

        self.predicted_sketch_action = []
        self.predicted_lf_action = []
        self.interaction_list = pack_utterance_to_interaction(interaction['interaction'])

        #[['physician'], ['department'], ['affiliated', 'with'], ['procedure'], ...]
        # self.table_names = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(' ')] for x in interaction['table_names']]

        # self.col_set = interaction['col_set']

        #['*', 'employee id', 'name', 'position', 'ssn', ..]
        # self.tab_cols = interaction['names']  # 列名

        # [-1, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2,...]
        # self.tab_ids = interaction['col_table']  # 表id
        # [['*'], ['employee', 'id'], ['name'], ['position'], ['ssn'], ...]
        # self.col_iter = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(" ")] for x in interaction['names']]

        #TODO ['*', 'regions . region id', 'regions . region name', ... 'job history . *', 'locations . *']
        self.column_names_embedder_input = interaction['columns_names_embedder']

        self.column_names_embedder_input_idxes = interaction['columns_names_embedder_idxes']
        # self.generate_column_names_embedder_input(interaction['names'],interaction['col_set'],interaction['col_table'],interaction['table_names'])


        self.num_col = len(self.column_names_embedder_input)

        self.schema_appear_mask = np.zeros((1,self.num_col),dtype=np.float32)


    def update_schema_appear_mask(self,idx):
        self.schema_appear_mask[0][idx] = 1
    def init_schema_appear_mask(self):
        self.schema_appear_mask = np.zeros((1,self.num_col),dtype=np.float32)        

    def previous_predicted_query(self,utterance_index):

        if utterance_index == 0:
            return [],[]
        else:
            return self.predicted_lf_action[utterance_index-1],self.predicted_sketch_action[utterance_index-1]


    def previous_query(self,utterance_index):

        if utterance_index == 0:
            return [],[]
        else:
            return self.interaction_list[utterance_index-1].target_actions,self.interaction_list[utterance_index-1].sketch_actions



    def set_column_name_embeddings(self, column_name_embeddings):
        self.column_name_embeddings = column_name_embeddings
        assert len(self.column_name_embeddings) == self.num_col

    def __len__(self):
        return len(self.interaction_list)

    def generate_column_names_embedder_input(self,column_names,column_set,table_idxes,table_names):
        column_names_embedder_input = []
        column_names_embedder_input_idxes = []
        # print(column_names)
        # col_set = set()
        # for i in column_names:
        #     col_set.add(i)

        for i, (table_id, column_name) in enumerate(zip(table_idxes,column_names)):
            if table_id >= 0:
                table_name = table_names[table_id]
                column_name_embedder_input = table_name + ' . ' + column_name
                column_name_embedder_input_idx = (table_id,column_set.index(column_name))
            else:
                #TODO *
                column_name_embedder_input = column_name
                column_name_embedder_input_idx = (-1,column_set.index(column_name))

            column_names_embedder_input.append(column_name_embedder_input)
            column_names_embedder_input_idxes.append(column_name_embedder_input_idx)
            # column_names_embedder_input_to_id[column_name_embedder_input] = len(self.column_names_embedder_input) - 1

        for i, table_name in enumerate(table_names):
            column_name_embedder_input = table_name + ' . *'
            column_names_embedder_input.append(column_name_embedder_input)
            column_name_embedder_input_idx = (i, column_set.index('*'))
            column_names_embedder_input_idxes.append(column_name_embedder_input_idx)


        for i, column_name in enumerate(column_set):
            if column_name!='*':
                column_name_embedder_input = column_name
                column_name_embedder_input_idx = (-1, i)
                column_names_embedder_input.append(column_name_embedder_input)
                column_names_embedder_input_idxes.append(column_name_embedder_input_idx)

        return column_names_embedder_input,column_names_embedder_input_idxes



# class UtteranceItem():
#     def __init__(self,utterance : dict):
#         pass


def pack_utterance_to_interaction(interaction : list):
    #ints = [
    #    UtteranceItem(inter,type) for inter in interaction
    #]
    ints = []
    previous_utter = None
    for cur_utter in interaction:
        ints.append(UtteranceItem(cur_utter,previous_utter))
        previous_utter = cur_utter
    return ints

class UtteranceItem():
    def __init__(self,utterance : dict,last_utter : dict,_type = 'train'):
        '''
        :param utterance: keys ->
                dict_keys(['utterance', 'nltk_pos', 'rule_label', 'utterance_toks', 'utterance_arg', 'sql',
                'utterance_arg_linking', 'query_toks_no_value', 'origin_utterance_toks', 'utterance_arg_type', 'query'])
        '''



        # self.sql_query = utterance['query']
        # self.utterance = utterance['utterance_toks']
        #TODO 要不要处理大小写，还是让bert处理？？
        # self.one_utterance =[wordnet_lemmatizer.lemmatize(x).lower() for x in utterance['utterance_toks']]
        #self.one_utterance = utterance['utterance_toks']
        #self.one_utterance_linking = utterance['utterance_arg_linking']
        
        
        col_cnt,tab_cnt = 0,0
        for x in utterance['utterance_arg_type']:
           if x[0] == 'table':
               tab_cnt+=1
           if x[0] == 'col':
               col_cnt += 1
        self.one_utterance = utterance['utterance_toks']
        self.one_utterance_linking = utterance['utterance_arg_linking']
        if last_utter == None:
            self.union_utterance = utterance['utterance_arg_linking']
        else:
            if tab_cnt!=0 and col_cnt!=0:
                 self.union_utterance = utterance['utterance_arg_linking']
            else:
                self.union_utterance = copy.deepcopy(last_utter['utterance_arg_linking'])
                self.union_utterance.extend(utterance['utterance_arg_linking'])
        
        
        if _type == 'train':
            self.target_actions = [eval(x) for x in utterance['rule_label'].strip().split(' ')]
            #for x in self.target_actions:
            #    print(type(x),x,x.id_c,'###',x.production,'###',x.parent)
            
            self.gold_query = utterance['query']

            self.sketch_actions = list()
            if self.target_actions:
                for action in self.target_actions:
                    if isinstance(action,define_rule.TC) or isinstance(action,define_rule.A) or \
                            isinstance(action,define_rule.Turn) or isinstance(action, define_rule.ValueS) or \
                            isinstance(action, define_rule.Distance):
                            continue
                    self.sketch_actions.append(action)

            self.action_num = len(self.target_actions)
            self.sketch_num  = len(self.sketch_actions)




    def __str__(self):
        text = 'utterance : {}\ngold query : {}\nrule_label : {}\n'.format(
            " ".join(x for x in self.utterance),self.gold_query,self.rule_label
        )
        return text
