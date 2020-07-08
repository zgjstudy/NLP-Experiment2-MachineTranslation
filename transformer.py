import tensorflow as tf

from bert import BertModelLayer
from bert.loader import StockBertConfig
from bert.loader import map_to_stock_variable_name

from utils import positional_encoding


def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = tf.cast(tf.shape(query)[-1], tf.float32)
    scores = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(d_k)
    if mask is not None:
        scores += (mask * -1e9)
    attention_weights = tf.nn.softmax(scores, axis=-1)
    return tf.matmul(attention_weights, value), attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.d_k = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.d_k))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # return (batch_size, num_heads, seq_len, d_k)

    def call(self, query, key, value, mask):
        batch_size = tf.shape(query)[0]
        query = self.split_heads(self.wq(query), batch_size)
        key = self.split_heads(self.wk(key), batch_size)
        value = self.split_heads(self.wv(value), batch_size)

        x, _ = scaled_dot_product_attention(query, key, value, mask)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (batch_size, -1, self.d_model))

        output = self.dense(x)
        return output


def position_wise_feed_forward(d_model, d_ff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(d_ff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, rate=0.1, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.masked_mha = MultiHeadAttention(d_model, num_heads)
        self.mha = MultiHeadAttention(d_model, num_heads)

        self.ffn = position_wise_feed_forward(d_model, d_ff)

        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, memory, target_mask, src_mask, training):
        out1 = self.masked_mha(x, x, x, target_mask)
        out1 = self.dropout1(out1, training)
        out1 = self.ln1(out1 + x)

        out2 = self.mha(out1, memory, memory, src_mask)
        out2 = self.dropout2(out2, training)
        out2 = self.ln2(out2 + out1)

        out3 = self.ffn(out2)
        out3 = self.dropout3(out3, training)
        out3 = self.ln3(out3 + out2)
        return out3


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, d_ff, target_vocab_size, rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(target_vocab_size, self.d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, d_ff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, memory, target_mask, src_mask, training):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, memory, target_mask, src_mask, training)

        return x  # (batch_size, target_seq_len, d_model)


def build_encoder(config_file):
    with tf.io.gfile.GFile(config_file, "r") as reader:
        stock_params = StockBertConfig.from_json_string(reader.read())
        bert_params = stock_params.to_bert_model_layer_params()

    return BertModelLayer.from_params(bert_params, name="bert")


class Transformer(tf.keras.Model):
    def __init__(self, decoder_config, bert_config_file, bert_training=False, rate=0.1):
        super(Transformer, self).__init__(name='transformer')

        self.encoder = build_encoder(config_file=bert_config_file)
        self.encoder.trainable = bert_training

        self.decoder = Decoder(decoder_config['num_layers'],
                               decoder_config['d_model'],
                               decoder_config['num_heads'],
                               decoder_config['d_ff'],
                               decoder_config['target_vocab_size'],
                               rate)

        self.final_layer = tf.keras.layers.Dense(decoder_config['target_vocab_size'])

    def load_stock_weights(self, bert_model, ckpt_file):
        ckpt_reader = tf.train.load_checkpoint(ckpt_file)

        bert_prefix = 'transformer/bert'

        weights = []
        for weight in bert_model.weights:
            stock_name = map_to_stock_variable_name(weight.name, bert_prefix)
            if ckpt_reader.has_tensor(stock_name):
                value = ckpt_reader.get_tensor(stock_name)
                weights.append(value)
            else:
                raise ValueError("No value for:[{}], i.e.:[{}] in:[{}]".format(
                    weight.name, stock_name, ckpt_file))
        bert_model.set_weights(weights)

    def restore_encoder(self, bert_ckpt_file):
        self.load_stock_weights(self.encoder, bert_ckpt_file)

    def call(self, inp, target, target_mask, src_mask, training):
        enc_output = self.encoder(inp, training=self.encoder.trainable)
        dec_output = self.decoder(target, enc_output, target_mask, src_mask, training)
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        return final_output
