#coding=utf8
# ========================================================
#   Copyright (C) 2021 All rights reserved.
#
#   filename : transformer.py
#   author   : liuzhiqiang1@360.cn
#   date     : 2021-08-17
#   desc     :
# ========================================================
import numpy as np
import tensorflow as tf

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_size, num_heads, attention_probs_dropout_prob, name=None):
        super(MultiHeadAttention, self).__init__(name=name)
        self.dim = hidden_size
        self.num_heads = num_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        assert hidden_size % num_heads == 0
        self.depth = self.dim // self.num_heads
        self.wq = tf.keras.layers.Dense(self.dim, kernel_initializer=self.create_initializer())
        self.wk = tf.keras.layers.Dense(self.dim, kernel_initializer=self.create_initializer())
        self.wv = tf.keras.layers.Dense(self.dim, kernel_initializer=self.create_initializer())
        self.wo = tf.keras.layers.Dense(self.dim, kernel_initializer=self.create_initializer())
        # Not mentioned in original transformer paper, however exists in google bert implemention
        self.attention_drop_layer = tf.keras.layers.Dropout(attention_probs_dropout_prob) # Not mentioned but exists

    def split_head(self, x, batch_size):
        '''
        split the last dimension (hidden_size) into (num_heads, depth)
        Parameters
        ----------
        x : [batch_size, seq_len, dim]
        batch_size : int
        Returns
        -------
        x_trans : [batch_size, num_head, seq_len, depth]
        '''
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=(0,2,1,3)) # [batch_size, num_head, seq_len, depth]

    def scaled_dot_product_attention(self, q, k, v, training=True, mask=None):
        '''
        calculate the attention weights and output
        Parameters
        ----------
        q : [batch_size, head_num, q_len, dim]
        k : [batch_size, head_num, kv_len, dim]
        v : [batch_dize, head_num, kv_len, dim]
        mask : padding mask [batch_size, 1, 1 kv_len]

        Returns
        output: [batch_size, q_len, dim]
        attention_weights: [batch_size, q_len, kv_len]
        '''
        matmul_qk = tf.matmul(q, k, transpose_b = True) # [batch_size, head_num, q_len, kv_len]
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.sqrt(dk)
        if mask is not None:
            scaled_attention_logits += (tf.cast(mask, tf.float32) * -1e9)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) # [batch_size, head_num, q_len, kv_len]
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_weights = self.attention_drop_layer(attention_weights, training)
        output = tf.matmul(attention_weights, v) # [batch_size, head_num, q_len, dim]
        return output, attention_weights

    def call(self, q, k, v, training=True, mask=None):
        '''
        Parameters
        ----------
        q : [batch_size, q_len, dim]
        k : [batch_size, kv_len, dim]
        v : [batch_dize, kv_len, dim]
        mask : padding mask [batch_size, 1, q_len, kv_len]

        Returns
        -------
        output: [batch_size, q_len, dim]
        attention_weights: []
        '''
        batch_size = tf.shape(q)[0]
        # linear layer and split into heads
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        split_q = self.split_head(q, batch_size)   # [batch_size, num_head, q_len, depth]
        split_k = self.split_head(k, batch_size)   # [batch_size, num_head, kv_len, depth]
        split_v = self.split_head(v, batch_size)   # [batch_size, num_head, kv_len, depth]
        scaled_attention, attention_weights = self.scaled_dot_product_attention(split_q, split_k, split_v, training, mask)
        # reshape
        concat_attention = tf.reshape(tf.transpose(scaled_attention, [0,2,1,3]), shape=(batch_size, -1, self.dim)) # [batch_size, q_len, dim]
        # final linear layer
        output = self.wo(concat_attention)
        return output, attention_weights

    def create_initializer(self):
        return tf.keras.initializers.TruncatedNormal(stddev=0.02)

    '''
    def get_config(self):
        config = {
                    "hidden_size": self.dim,
                    "num_heads": self.num_heads,
                    "attention_probs_dropout_prob": self.attention_probs_dropout_prob
                 }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    '''


class TransformerEncoderBlock(tf.keras.layers.Layer):
    # A transformer encoder block
    # MultiHeadAtt --> Dropout ---> Add --> LN --> FFN --> Dropout ---> Add --> LN
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, hidden_dropout_prob, attention_probs_dropout_prob, name=None):
        super(TransformerEncoderBlock, self).__init__(name=name)
        self.dim = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.MultiHeadAttLayer = MultiHeadAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.hidden_drop_layer1 = tf.keras.layers.Dropout(hidden_dropout_prob)
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dense1 = tf.keras.layers.Dense(intermediate_size, activation='gelu', kernel_initializer=self.create_initializer())
        self.dense2 = tf.keras.layers.Dense(hidden_size, kernel_initializer=self.create_initializer())
        self.hidden_drop_layer2 = tf.keras.layers.Dropout(hidden_dropout_prob)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def activate(self):
        """Gaussian Error Linear Unit.
        This is a smoother version of the RELU. already builtin in tensorflow
        Original paper: https://arxiv.org/abs/1606.08415
        Args:
            x: float Tensor to perform activation.
        Returns:
            `x` with the GELU activation applied.
        """
        def gelu(x):
            cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
            return x * cdf

        return gelu

    def call(self, inputs, training=True, mask=None):
        attn_output, atten_weights = self.MultiHeadAttLayer(inputs, inputs, inputs, training, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.hidden_drop_layer1(attn_output, training=training)
        out1 = self.layer_norm1(inputs + attn_output)  # [batch_size, input_seq_len, d_model]
        ffn_output = self.dense1(out1)                 # intermediate ffn layer with gelu activation
        ffn_output = self.dense2(ffn_output)
        ffn_output = self.hidden_drop_layer2(ffn_output, training=training)
        out2 = self.layer_norm2(out1 + ffn_output)     # (batch_size, input_seq_len, d_model)
        return out2, atten_weights

    def create_initializer(self):
        return tf.keras.initializers.TruncatedNormal(stddev=0.02)
 
    '''
    def get_config(self):
        config = {
                    "hidden_size": self.dim,
                    "num_attention_heads": self.num_attention_heads,
                    "intermediate_size": self.intermediate_size,
                    "intermediate_act_fn": self.intermediate_act_fn,
                    "hidden_dropout_prob": self.hidden_dropout_prob,
                    "attention_probs_dropout_prob": self.attention_probs_dropout_prob
                 }
        base_config = super(TransformerEncoderBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    '''


if __name__ == "__main__":
    batch_size = 3
    seq_len = 10
    dim = 8
    x   = tf.random.uniform(shape=[batch_size, seq_len, dim], dtype=tf.float32)
    msk = tf.random.uniform(shape = [batch_size,1], minval = 2, maxval = 3, dtype = tf.int32)       #[batch_size, 1]
    msk = tf.cast(tf.logical_not(tf.sequence_mask(msk, seq_len)), dtype=tf.float32)                 #[batch_size, 1, seq_len]
    msk = tf.expand_dims(msk, 1)                                                                    #[batch_size, 1, 1, seq_len]
    mha = MultiHeadAttention(16, 2, 0.0)
    y1, y2 = mha(x, x, x, mask = msk)
    teb = TransformerEncoderBlock(8, 2, 256, 0.1, 0.1)
    out, atten_wei = teb(x, training=True, mask=msk)
    print(out)
    print(atten_wei)
