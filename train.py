# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# -*- coding: utf-8 -*-
"""
# @Time    : 2019/12/09
# @Author  : xiaxia.wang
# @File    : data_process.py
# @Software: PyCharm
"""

import time
import traceback

import os
import torch
import torch.optim as optim
import tqdm
import copy,json
from logger import Logger

import torch.nn as nn
import parse_args_local_pointer as args
from src import utils
from src.models.model_decoder_subtree_pointer2 import EditTreeNet
from src.rule import semQL
from data_util import sparc_data


def train(args,model,optimizer,bert_optimizer,data):
    '''
    :param args:
    :param model:
    :param data:
    :param grammar:
    :return:
    '''

    print('Loss epoch threshold: %d' % args.loss_epoch_threshold)
    print('Sketch loss coefficient: %f' % args.sketch_loss_coefficient)

    if args.load_model and not args.resume:
        print('load pretrained model from %s'% (args.load_model))
        pretrained_model = torch.load(args.load_model,map_location=lambda storage, loc: storage)
        pretrained_modeled = copy.deepcopy(pretrained_model)
        for k in pretrained_model.keys():
            if k not in model.state_dict().keys():
                del pretrained_modeled[k]

        model.load_state_dict(pretrained_modeled)

    # ==============data==============
    if args.interaction_level:
        batch_size = 1
        train_batchs, train_sample_batchs = data.get_interaction_batches(batch_size,
                                                                         sample_num=args.train_evaluation_size,
                                                                         use_small=args.toy)
        valid_batchs = data.get_all_interactions(data.val_sql_data, data.val_table_data,_type='test',use_small=args.toy)

    else:
        batch_size = args.batch_size
        train_batchs = data.get_utterance_batches(batch_size)
        valid_batchs = data.get_all_utterances(data.val_sql_data)
    print(len(train_batchs),len(train_sample_batchs),len(valid_batchs))
    start_epoch = 1
    best_question_match = .0
    lr = args.initial_lr
    stage = 1

    if args.resume:
        model_save_path = utils.init_log_checkpoint_path(args)
        current_w = torch.load(os.path.join(model_save_path, args.current_model_name))
        best_w = torch.load(os.path.join(model_save_path, args.best_model_name))
        best_question_match = best_w['question_match']
        start_epoch = current_w['epoch'] + 1
        lr = current_w['lr']
        utils.adjust_learning_rate(optimizer, lr)
        stage = current_w['stage']
        model.load_state_dict(current_w['state_dict'])
        # 如果中断点恰好为转换stage的点
        if start_epoch - 1 in args.stage_epoch:
            stage += 1
            lr /= args.lr_decay
            utils.adjust_learning_rate(optimizer, lr)
            model.load_state_dict(best_w['state_dict'])
        print("=> Loading resume model from epoch {} ...".format(start_epoch - 1))

    # model.word_emb = utils.load_word_emb(args.glove_embed_path,use_small=args.use_small)
    # begin train

    model_save_path = utils.init_log_checkpoint_path(args)
    utils.save_args(args, os.path.join(model_save_path, 'config.json'))
    file_mode = 'a' if args.resume else 'w'
    log = Logger(os.path.join(args.save, args.logfile), file_mode)
    # log_pred_gt = Logger(os.path.join(args.save, args.log_pred_gt), file_mode)

    with open(os.path.join(model_save_path, 'epoch.log'), 'w') as epoch_fd:

        for epoch in range(start_epoch,args.epoch + 1):
            epoch_begin = time.time()
            # model.set_dropout(args.dropout_amount)

            model.dropout_ratio = args.dropout_amount

            if args.interaction_level:
                loss = utils.epoch_train_with_interaction(epoch,log,model,optimizer,bert_optimizer,train_batchs,args)
                #loss = 2.
            else:
                pass
            
            model.dropout_ratio = 0.
            # model.set_dropout(0.)
            epoch_end = time.time()
                    
            s = time.time()
            sample_sketch_acc, sample_lf_acc, sample_interaction_lf_acc = utils.epoch_acc_with_interaction(epoch,model,train_sample_batchs, args,
                                                                                                     beam_size=args.beam_size,
                                                                                                     use_predicted_queries=True)
            log_str = '[Epoch: %d(sample predicted),Sample ratio[%d]: %f], Sketch Acc: %f, Acc: %f, Interaction Acc: %f, Train time: %f, Sample predict time: %f\n' % (
                epoch,len(train_sample_batchs), len(train_sample_batchs) / len(train_batchs),sample_sketch_acc, sample_lf_acc, sample_interaction_lf_acc, epoch_end - epoch_begin,time.time()-s)
            print(log_str)
            log.put(log_str)
            epoch_fd.write(log_str)
            epoch_fd.flush()
            
            # s = time.time()
            #
            # gold_sketch_acc,gold_lf_acc ,gold_interaction_lf_acc = utils.epoch_acc_with_interaction(epoch,model, valid_batchs,args,beam_size=args.beam_size)
            #
            # log_str = '[Epoch: %d(gold)], Loss: %f, Sketch Acc: %f, Acc: %f, Interaction Acc: %f, Gold predict time: %f\n' % (
            #     epoch, loss, gold_sketch_acc, gold_lf_acc, gold_interaction_lf_acc,time.time()-s)
            # print(log_str)
            # log.put(log_str)
            # epoch_fd.write(log_str)
            # epoch_fd.flush()
            
            s = time.time()
            valid_jsonf , pred_sketch_acc, pred_lf_acc, pred_interaction_lf_acc = utils.epoch_acc_with_interaction_save_json(epoch,model, valid_batchs, args,
                                                                                      beam_size=args.beam_size,
                                                                                      use_predicted_queries=True)

            question_match,interaction_match = utils.semQL2SQL_question_and_interaction_match(valid_jsonf,args)

            log_str = '[Epoch: %d(predicted)], Loss: %f, lr: %.3ef, Sketch Acc: %f, Acc: %f, Interaction Acc: %f, Question Match : %f, Interaction Macth : %f, Predicted predict time: %f\n\n' % (
                epoch, loss,optimizer.param_groups[0]["lr"], pred_sketch_acc, pred_lf_acc, pred_interaction_lf_acc,question_match,interaction_match ,time.time()-s)
            print(log_str)
            log.put(log_str)
            epoch_fd.write(log_str)
            epoch_fd.flush()


            state = {"state_dict": model.state_dict(), "epoch": epoch,
                     "question_match": question_match, "interaction_match": interaction_match,
                     "lr": lr, 'stage': stage}

            current_w_name = os.path.join(model_save_path, '{}_{:.3f}.pth'.format(epoch, question_match))
            best_w_name = os.path.join(model_save_path, args.best_model_name)
            current_w_name2 = os.path.join(model_save_path, args.current_model_name)

            utils.save_ckpt(state, best_question_match < question_match, current_w_name, best_w_name,current_w_name2)
            best_question_match = max(best_question_match, question_match)
            if epoch in args.stage_epoch:
                stage += 1
                lr /= args.lr_decay
                best_w_name = os.path.join(model_save_path, args.best_model_name)
                model.load_state_dict(torch.load(best_w_name)['state_dict'])
                print("*" * 10, "step into stage%02d lr %.3ef" % (stage, lr))
                utils.adjust_learning_rate(optimizer, lr)

    
    log.put("Finished training!")
    log.close()
    # log_pred_gt.close()


def build_optim(args,model):
    params_trainer = []
    params_bert_trainer = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'model_bert' in name:
                params_bert_trainer.append(param)
            else:
                params_trainer.append(param)
    trainer = optim.Adam(params_trainer, lr=args.initial_lr)
    if args.fine_tune_bert:
        bert_trainer = optim.Adam(params_bert_trainer, lr=args.lr_bert)
    return trainer,bert_trainer


if __name__ == '__main__':
    # arg_param = args.interpret_args()
    #TODO local debug
    args_param = args.get_local_args()
    args = args.init_config(args_param)

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    grammar = semQL.Grammar()

    #data
    data = sparc_data.SparcDataset(args)
    #model
    model = EditTreeNet(args, grammar)
    model.to(device)
    if torch.cuda.device_count()>1:
        model = nn.DataParallel(model)
    #optimizer
    optimizer,bert_optimizer = build_optim(args,model)

    train(args,model,optimizer,bert_optimizer,data)
