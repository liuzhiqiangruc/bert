#coding=utf8
# ========================================================
#   Copyright (C) 2021 All rights reserved.
#   
#   filename : model.py

#   author   : ***
#   date     : 2021-08-18
#   desc     : 
# ======================================================== 
import os
import sys
import datetime
import tensorflow as tf
import numpy as np
from transformer import TransformerEncoderBlock
from conf import *

'''
input paramenters define
'''
input_ids = tf.TensorSpec(shape=(None, TRAIN_MAX_SEQ_LEN), dtype=tf.string, name="input_ids")
valid_len = tf.TensorSpec(shape=(None, ), dtype = tf.int32, name = "valid_len")
input_spec = tf.tuple([input_ids, valid_len])

class Bert(tf.keras.Model):
    def __init__(self, block_count, hidden_size, n_header, intermediate_size, hidden_drop_prob, att_drop_prob, id_mapping_file, cdf, neg_size, max_voc_size, name = None):
        super(Bert, self).__init__(name = name)
        self._block_count = block_count                     # transformers num
        self._hidden_size = hidden_size                     # hiddensize for transformeroutput and embeddinginput which need same
        self._n_header    = n_header                        # number of headers
        self._max_voc_sz  = max_voc_size                    # max size for word(url) embedding
        self._inner_h_sz  = intermediate_size               # hiddensize for transformer_ffn_mediate layer
        self._h_drop_prob = hidden_drop_prob                # drop_prob for hiddenoutput of transformers
        self._a_drop_prob = att_drop_prob                   # drop_prob for attentation matrix
        self._id_map_file = id_mapping_file                 # name_id_mapping file
        self._cdf         = cdf                             # cumsum prob for neg sample
        self._neg_size    = neg_size                        # neg sample count for each pos target
        self._id_m_layer  = self._create_id_map_layer()     # id-mapper layer for name lookup
        self._embedding   = self._create_emb()              # url embedding layers
        self._transforms  = self._create_transformers()     # transformers

    def _create_transformers(self):
        return [TransformerEncoderBlock(self._hidden_size, 
                                        self._n_header, 
                                        self._inner_h_sz, 
                                        self._h_drop_prob, 
                                        self._a_drop_prob,
                                        name = "tb_%d" %i) for i in range(self._block_count)]

    def _create_id_map_layer(self):
        tfinit = tf.lookup.TextFileInitializer(self._id_map_file, key_index = 0, value_index = 1, key_dtype=tf.string, value_dtype=tf.int32, delimiter=",")
        return tf.lookup.StaticHashTable(tfinit, default_value = 0)

    def _create_emb(self):
        _emb = tf.keras.layers.Embedding(self._max_voc_sz, self._hidden_size)
        _emb.build([None, self._hidden_size])
        _emb.set_weights([tf.random.truncated_normal(shape=[self._max_voc_sz, self._hidden_size], stddev=0.02)])
        return _emb

    @tf.function
    def call(self, x):
        '''
        parse input x: [[seqids],[labelindex],[labels],[validseq&labellength]]
        '''
        bsize    = tf.shape(x[0])[0]                    # batch_size    rank:0
        seqids   = x[0]                                 # input seq all    [batch_size, 30]
        lbl_idx  = x[1]                                 # label index      [batch_size, 4]
        lbl_ids  = x[2]                                 # label ids        [batch_size, 4]
        seq_len  = tf.shape(seqids)[1]                  # max seq length
        lbl_len  = tf.shape(lbl_ids)[1]                 # max lbl length
        seq_vlen = x[3][:, 0]                           # valid seq length [batch_size, ]
        lbls_c   = x[3][:, 1]                           # valid lbl length [batch_size, ]
        ''' generate self-attention encode embedding for given inputs '''
        emb = self._embedding(seqids)                      # [batchsize, seqlen, hidden_size]
        seq_msk  = tf.cast(tf.logical_not(tf.sequence_mask(seq_vlen, seq_len)), dtype = tf.float32)
        seq_msk  = tf.expand_dims(seq_msk, axis = 1)
        seq_msk  = tf.expand_dims(seq_msk, axis = 1)
        for i in range(self._block_count):
            emb, _ = self._transforms[i](emb, mask = seq_msk)
        semb    = tf.reshape(emb, shape = [-1, self._hidden_size])
        lbl_idx = tf.reshape(tf.range(bsize) * seq_len, shape = [-1, 1]) + lbl_idx
        lbl_emb = tf.gather(semb, lbl_idx)                 # [batch, 4, dim]
        lbl_emb = tf.expand_dims(lbl_emb, axis = 2)        # [batch, 4, 1, dim] same shape with output labels
        ''' negitive sample and get output embedding '''
        rnd_val = tf.random.uniform([bsize * lbl_len * self._neg_size], minval = 0, maxval = 1.0)
        neg_ids = tf.reshape(tf.searchsorted(self._cdf, rnd_val, side='left'), shape = [bsize, lbl_len, self._neg_size])
        lbl_ids = tf.expand_dims(lbl_ids, axis = -1)       # [batch, 4, 1]
        tar_ids = tf.concat([lbl_ids, neg_ids], axis = -1) # [batch, 4, negsize + 1]
        tar_emb = self._embedding(tar_ids)                 # [batch, 4, negsize + 1, dim]
        tar_emb = tf.math.l2_normalize(tar_emb, axis = -1)
        ''' logit & probs using inner product with transform encode and output embedding(shared with input embedding layer) '''
        logit   = tf.reduce_sum(tf.multiply(lbl_emb, tar_emb), axis = -1) # [batch, 4, negsize+1]
        probs   = tf.nn.log_softmax(logit, axis = -1)
        probs   = probs[:,:,0]                             # [batch, 4] just keep positive label probs for loss calculation
        ''' mask labels '''
        lbl_wei = tf.cast(tf.sequence_mask(lbls_c, lbl_len), dtype = tf.float32)
        loss    = tf.reduce_sum(tf.multiply(probs, lbl_wei)) / tf.reduce_sum(lbl_wei)
        return -loss

#   @tf.function
#   def call(self, x):
#       '''
#       parse input x: [[seqids],[labelindex],[labels],[validseq&labellength]]
#       '''
#       bsize    = tf.shape(x[0])[0]                       # batch_size    rank:0
#       seqids   = x[0]                                    # input seq all    [batch_size, 30]
#       lbl_idx  = x[1]                                    # label index      [batch_size, 4]
#       lbl_ids  = x[2]                                    # label ids        [batch_size, 4]
#       seq_len  = tf.shape(seqids)[1]                     # max seq length
#       lbl_len  = tf.shape(lbl_ids)[1]                    # max lbl length
#       seq_vlen = x[3][:, 0]                              # valid seq length [batch_size, ]
#       lbls_c   = x[3][:, 1]                              # valid lbl length [batch_size, ]
#       ''' generate self-attention encode embedding for given inputs '''
#       emb = self._embedding(seqids)                      # [batchsize, seqlen, hidden_size]
#       seq_msk  = tf.cast(tf.logical_not(tf.sequence_mask(seq_vlen, seq_len)), dtype = tf.float32)
#       seq_msk  = tf.expand_dims(seq_msk, axis = 1)
#       seq_msk  = tf.expand_dims(seq_msk, axis = 1)
#       for i in range(self._block_count):
#           emb, _ = self._transforms[i](emb, mask = seq_msk)
#       semb    = tf.reshape(emb, shape = [-1, self._hidden_size])
#       lbl_idx = tf.reshape(tf.range(bsize) * seq_len, shape = [-1, 1]) + lbl_idx
#       lbl_emb = tf.gather(semb, lbl_idx)                 # [batch, 4, dim]
#       lbl_emb = tf.expand_dims(lbl_emb, axis = 2)        # [batch, 4, 1, dim] same shape with output labels
#       ''' negitive sample and get output embedding '''
#       rnd_val = tf.random.uniform([bsize * lbl_len * self._neg_size], minval = 0, maxval = 1.0)
#       neg_ids = tf.reshape(tf.searchsorted(self._cdf, rnd_val, side='left'), shape = [bsize, lbl_len, self._neg_size])
#       lbl_ids = tf.expand_dims(lbl_ids, axis = -1)       # [batch, 4, 1]
#       tar_ids = tf.concat([lbl_ids, neg_ids], axis = -1) # [batch, 4, negsize + 1]
#       tar_emb = self._embedding(tar_ids)                 # [batch, 4, negsize + 1, dim]
#       #tar_emb = tf.math.l2_normalize(tar_emb, axis = -1)
#       ''' logit & probs using inner product with transform encode and output embedding(shared with input embedding layer) '''
#       logit   = tf.reduce_sum(tf.multiply(lbl_emb, tar_emb), axis = -1) # [batch, 4, negsize+1]
#       #probs   = tf.nn.log_softmax(logit, axis = -1)
#       #probs   = probs[:,:,0]                             # [batch, 4] just keep positive label probs for loss calculation
#       pos     = tf.ones([bsize, lbl_len, 1], dtype = tf.float32)
#       neges   = tf.zeros([bsize, lbl_len, self._neg_size], dtype = tf.float32)
#       labels  = tf.concat([pos, neges], axis = -1)
#       losses  = tf.nn.sigmoid_cross_entropy_with_logits(labels, logit)
#       probs   = tf.reduce_mean(losses, axis = -1)
#       ''' mask labels '''
#       lbl_wei = tf.cast(tf.sequence_mask(lbls_c, lbl_len), dtype = tf.float32)
#       loss    = tf.reduce_sum(tf.multiply(probs, lbl_wei)) / tf.reduce_sum(lbl_wei)
#       ploss   = tf.reduce_sum(tf.multiply(losses[:,:,0], lbl_wei)) / tf.reduce_sum(lbl_wei)
#       nloss   = tf.reduce_sum(tf.multiply(tf.reduce_mean(losses[:,:,1:], axis = -1), lbl_wei)) / tf.reduce_sum(lbl_wei)
#       return loss, ploss, nloss

    @tf.function(input_signature=[input_spec])
    def masked_emb_inference(self, input_ids):
        seqids = input_ids[0]
        vallen = input_ids[1]
        spe    = tf.shape(seqids)
        batch  = spe[0]
        length = spe[1]
        ids    = self._id_m_layer.lookup(seqids)
        ones   = tf.ones([batch, 1], dtype = tf.int32)
        ids    = tf.concat([ones, ids], axis = -1)[:,0:length]
        vallen = tf.math.minimum(vallen + 1, length)
        emb    = self._embedding(ids)
        mask   = tf.cast(tf.logical_not(tf.sequence_mask(vallen, length)), dtype = tf.float32)
        mask   = tf.expand_dims(mask, axis = 1)
        mask   = tf.expand_dims(mask, axis = 1)
        for i in range(self._block_count):
            emb, att = self._transforms[i](emb, training = False, mask = mask)
        emb    = emb[:,0,:]
        return emb, att

    @tf.function(input_signature=[input_spec])
    def avg_emb_inference(self, input_ids):
        seqids = input_ids[0]
        vallen = input_ids[1]
        length = tf.shape(seqids)[1]
        ids    = self._id_m_layer.lookup(seqids)
        emb    = self._embedding(ids)
        mask   = tf.cast(tf.logical_not(tf.sequence_mask(vallen, length)), dtype = tf.float32)
        mask   = tf.expand_dims(mask, axis = 1)
        mask   = tf.expand_dims(mask, axis = 1)
        for i in range(self._block_count):
            emb, att = self._transforms[i](emb, training = False, mask = mask)
        mask   = tf.cast(tf.sequence_mask(vallen, length), dtype = tf.float32)
        emask  = tf.expand_dims(mask, axis = -1)
        emb    = tf.reduce_sum(tf.multiply(emb, emask), axis = 1) / tf.expand_dims(tf.reduce_sum(mask, axis = -1), axis = -1)
        return emb, att



class Trainer():
    def __init__(self, input_file, dict_file, max_length, max_labels, shuffle_size, batch_size, label_rate, max_epoch, parallels, power):
        '''
        input file format: uid \t url \t url \t url....
                           we do not use uid during training process
        dict  file : word dict index file, which incr from 2, 0 as default, 1 as [CLS]
        '''
        self._input_     = input_file
        self._ids_train  = "%s_ids" %(input_file)
        self._lbl_rate   = int(label_rate * 100)
        self._dict_file  = dict_file
        self._max_length = max_length
        self._max_labels = max_labels
        self._sf_size    = shuffle_size
        self._batch_size = batch_size
        self._max_epoch  = max_epoch
        self._p_         = parallels                    # num_parallel_reads
        self._power      = power
        self._cdf, self._max_v_sz = self.update_dict()
        if self._max_v_sz < 1048576:
            self._max_v_sz = 1048576
        self._model      = Bert (BERT_DEEP, BERT_DIM
                                ,BERT_HEADER
                                ,BERT_INNER_DIM
                                ,BERT_H_DROP_PROB
                                ,BERT_A_DROP_PROB
                                ,self._dict_file
                                ,self._cdf
                                ,BERT_NEG_SIZE
                                ,max_voc_size = self._max_v_sz )

    def update_dict(self):
        '''
        update dict {word:idx} using new train file
        and generate train_input with id tokens
        '''
        voc_dict = {}
        word_cnt = {}
        maxid = 1
        with open (self._dict_file, "r") as fp:
            for line in fp:
                word, idx = line.strip().split(",")
                idx = int(idx)
                if maxid < idx:
                    maxid = idx
                voc_dict[word] = idx
        maxid += 1
        with open(self._input_, "r") as ifp, open(self._ids_train, "w") as ofp:
            for line in ifp:
                segs = line.strip().split("\t")
                for url in segs:
                    if not url in voc_dict:
                        voc_dict[url] = maxid
                        maxid += 1
                    word_cnt.setdefault(url, 0)
                    word_cnt[url] += 1
                ids = [voc_dict[x] for x in segs]
                ofp.write("%s\n" %(",".join(map(str, ids))))
        pdf = [0 for i in range(maxid)]
        for word, cnt in word_cnt.items():
            pdf[voc_dict[word]] = cnt
        temp_file = "%s_tmp" %(self._dict_file)
        sl = sorted(voc_dict.items(), key = lambda x: x[1])
        with open(temp_file, "w") as ofp:
            for url, idx in sl:
                ofp.write("%s,%d\n" %(url, idx))
        cmd = "mv %s_tmp %s" %(self._dict_file, self._dict_file)
        ret = os.system(cmd)
        assert (ret == 0)
        pdf = tf.math.pow(tf.constant(pdf, dtype = tf.float32), self._power)
        pdf = pdf / tf.reduce_sum(pdf)
        cdf = tf.math.cumsum(pdf)
        return cdf, maxid

    def load_dataset(self):
        def _parse_line(line):
            ids = tf.strings.to_number(tf.strings.split(tf.strings.strip(line), ','), out_type=tf.dtypes.int32)
            ids = ids[0 : self._max_length]
            isp = tf.shape(ids)
            ict = tf.math.minimum(tf.math.maximum(tf.cast(isp * self._lbl_rate / 100, dtype = tf.int32), 1), self._max_labels)
            idx = tf.random.shuffle(tf.range(isp[0]))[0:ict[0]]
            spd = tf.gather(ids, idx)
            ons = tf.ones(ict, dtype = tf.dtypes.int32)
            ids = tf.tensor_scatter_nd_update(ids, tf.expand_dims(idx, axis = -1), ons)
            pad = tf.zeros(self._max_length, dtype=tf.dtypes.int32)
            ids = tf.concat([ids, pad], axis = -1)[0:self._max_length]
            idx = tf.concat([idx, pad], axis = -1)[0:self._max_labels]
            spd = tf.concat([spd, pad], axis = -1)[0:self._max_labels]
            szs = tf.concat([isp, ict], axis = -1)
            ret = tf.tuple([ids, idx, spd, szs])     # [inputs, labelindex, labels, valid_input_label_length]
            return ret
        ds = tf.data.TextLineDataset(self._ids_train, num_parallel_reads = self._p_)  \
                .map(_parse_line, num_parallel_calls = self._p_)      \
                .shuffle(self._sf_size)                               \
                .batch(self._batch_size)                              \
                .prefetch(tf.data.experimental.AUTOTUNE)
        return ds

    def __call__ (self):
        optimizer = tf.keras.optimizers.Adagrad(learning_rate = TRAIN_LEARNING_RATE)

        @tf.function
        def _single_train_step(x):
            with tf.GradientTape() as tape:
                loss = self._model(x)
                grad = tape.gradient(loss, self._model.trainable_variables)
                optimizer.apply_gradients(grads_and_vars=zip(grad, self._model.trainable_variables))
            return loss

        train_ds = self.load_dataset()
        batch = 0
        for epoch in range(self._max_epoch):
            losses = []
            for x in train_ds:
                batch += 1
                loss = _single_train_step(x)
                losses.append(loss.numpy())
                if batch % 100 == 0:
                    print("batch: %d,  loss: %.6f" %(batch, np.mean(losses)), datetime.datetime.now(), file = sys.stderr)
                    sys.stderr.flush()
                    losses = []
            print("batch: %d,  loss: %.6f" %(batch, np.mean(losses)), datetime.datetime.now(), file = sys.stderr)
            print("************************* epoch done *******************************", file = sys.stderr)
            sys.stderr.flush()
            self._model.save(MODEL_OUTPUT_DIR, save_format = 'tf')
        print("train process done", file = sys.stderr)
        sys.stderr.flush()
        self._model.save(MODEL_OUTPUT_DIR, save_format = 'tf')
        return 0

if __name__ == "__main__":
    input_train = sys.argv[1]
    dict_file   = sys.argv[2]
    train = Trainer(input_train, dict_file
                        ,TRAIN_MAX_SEQ_LEN
                        ,TRAIN_MAX_LBL_LEN
                        ,TRAIN_SHUFFLE_SIZE
                        ,TRAIN_BATCH_SIZE
                        ,TRAIN_LABEL_RATE
                        ,TRAIN_MAX_EPOCH
                        ,TRAIN_PARALLELS
                        ,TRAIN_NEG_POWER )
    train()
