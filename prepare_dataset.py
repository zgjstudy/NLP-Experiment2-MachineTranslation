import tensorflow as tf

import config
from tokenizer import tokenize


class Gen_data:
    def __init__(self, zh, en):
        self.zh = zh
        self.en = en

    def __iter__(self):
        for data in zip(self.zh, self.en):
            yield data


def filter_rl(zh, en):
    ratio = len(zh) / len(en)
    return tf.logical_and(tf.logical_and(0.5 <= ratio, ratio <= 5),
                          tf.logical_and(tf.size(zh) <= config.max_seq_len, tf.size(en) <= config.max_seq_len))


def make_dataset(global_batch_size):
    zh_ids, en_ids, tokenizer_en, tokenizer_zh = tokenize()

    train_data_size = int(len(zh_ids) * 0.7)
    train_zh_ids = zh_ids[:train_data_size]
    val_zh_ids = zh_ids[train_data_size:]
    train_en_ids = en_ids[:train_data_size]
    val_en_ids = en_ids[train_data_size:]

    train_dataset = tf.data.Dataset.from_generator(Gen_data(train_zh_ids, train_en_ids).__iter__, (tf.int32, tf.int32))
    train_dataset = train_dataset.filter(filter_rl)
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(2000).padded_batch(
        global_batch_size, padded_shapes=([config.max_seq_len], [config.max_seq_len]), drop_remainder=True)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_generator(Gen_data(val_zh_ids, val_en_ids).__iter__, (tf.int32, tf.int32))

    train_steps_per_epoch = train_data_size // global_batch_size
    val_steps = len(val_zh_ids)
    return train_dataset, val_dataset, tokenizer_zh, tokenizer_en, train_steps_per_epoch, val_steps
