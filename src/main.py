#coding=utf8
# ========================================================
#   Copyright (C) 2021 All rights reserved.
#   
#   filename : main.py
#   author   : ***
#   date     : 2021-09-02
#   desc     : 
# ======================================================== 
import os
import sys
import datetime, time
import tensorflow as tf
import numpy as np
from transformer import TransformerEncoderBlock
from model import Trainer
from conf  import *
from prepare import process_train_asto_md5s

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
physical_devices = tf.config.list_physical_devices('GPU')
try:
    for x in physical_devices:
        tf.config.experimental.set_memory_growth(x, True)
except:
    pass

if __name__ == "__main__":
    process_train_asto_md5s(TRAIN_INPUT)
    input_md5s = "%s_md5s" %TRAIN_INPUT
    train = Trainer(input_md5s, TRAIN_DICT 
                        ,TRAIN_MAX_SEQ_LEN
                        ,TRAIN_MAX_LBL_LEN
                        ,TRAIN_SHUFFLE_SIZE
                        ,TRAIN_BATCH_SIZE
                        ,TRAIN_LABEL_RATE
                        ,TRAIN_MAX_EPOCH
                        ,TRAIN_PARALLELS
                        ,TRAIN_NEG_POWER )
    train()
