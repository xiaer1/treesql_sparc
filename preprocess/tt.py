import nltk
from preprocess.utils_source import AGG, wordnet_lemmatizer,symbol_filter

question = 'what is model name with most make ?'
# question = 'what is lowest accelerate for any car'
# question = 'give length of match in minute of shortest match'
# question = 'what is birth date of oldest player ?'
# question = 'how about maximum ?'
# question = 'which department has most employee ? give me department name .'
# question = 'which of those refer to oldest employee ?'
# question = 'for department with fewest professor , what is it name and code ?'
# question = 'which department has most employee ? give me department name .'
# q  = symbol_filter(question.split(' '))
# print(q)
# question_toks = [wordnet_lemmatizer.lemmatize(x.lower()) for x in q if x.lower() != 'the']
# print(question_toks)
# nltk_result = nltk.pos_tag(question_toks)
# print(nltk_result)


# question = "Intersect that with last name Sawayn and nickname containing letter 's'."
# question = [['intersect'], ['that'], ['with'], ['last', 'name'], ['sawayn'], ['and'], ['nickname'], ['containing'], ['letter'], ["'s"], ["'"], ['.'], ['1']]
# m = [' '.join(x) for x in question ]
# t = symbol_filter(m)
# print(t)
# from preprocess.data_process_sparc import convert_time_format
# question = ['which', 'one', 'oct', '23th', ',', '2010', '?']
# # q = [' '.join(x) for x in question]
# convert_time_format(question,question)
# import eventlet
# eventlet.monkey_patch()
# import time
# time_limit = 1
# with eventlet.Timeout(time_limit, True):
#     while(True):
#         print('error-->')
#
# print('error')

# from __future__ import with_statement # Required in 2.5
import signal
from contextlib import contextmanager

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException( "Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
import sqlite3
import os
def func():

    db = '../sparc/database/car_1/car_1.sqlite'
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    print(db)


    p_str = 'SELECT T1.CountryName FROM cars_data AS T2 JOIN car_names AS T3 JOIN model_list AS T4 JOIN car_makers AS T5 ' \
            'JOIN countries AS T1 WHERE T2.Weight < ( SELECT avg ( T3.Weight ) FROM cars_data AS T3 )'
    cursor.execute(p_str)
    p_res = cursor.fetchall()
try:
    with time_limit(1):
       func()
except TimeoutException as e:
    print("Timed out2222!")
except Exception as e:
    print("Error!")