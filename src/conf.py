#coding=utf8
# ========================================================
#   Copyright (C) 2021 All rights reserved.
#   
#   filename : conf.py
#   author   : ***
#   date     : 2021-08-29
#   desc     : 
# ======================================================== 


CUDA_VISIBLE_DEVICES = "1"

# prepare corpus for train
MIN_SESSION_LENGTH  = 3

# params for train process
TRAIN_LEARNING_RATE = 0.01
TRAIN_SHUFFLE_SIZE  = 1000000
TRAIN_BATCH_SIZE    = 1024
TRAIN_LABEL_RATE    = 0.15
TRAIN_MAX_EPOCH     = 30
TRAIN_PARALLELS     = 10
TRAIN_NEG_POWER     = 0.6
TRAIN_MAX_SEQ_LEN   = 30
TRAIN_MAX_LBL_LEN   = 4
TRAIN_INPUT         = "../data/train"
TRAIN_DICT          = "../data/dict"
MODEL_OUTPUT_DIR    = "../model_out/model.new"


# params for bert model
BERT_DEEP           = 4
BERT_DIM            = 256
BERT_HEADER         = 8
BERT_INNER_DIM      = 1024
BERT_H_DROP_PROB    = 0.1
BERT_A_DROP_PROB    = 0.1
BERT_NEG_SIZE       = 10
