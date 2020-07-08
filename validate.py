import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from tqdm import trange
import tensorflow as tf

import config
from prepare_dataset import make_dataset
from transformer import Transformer
from translator import Translator
from utils import CustomSchedule

assert tf.__version__.startswith('2.')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

transformer = Transformer(config.dec_config, config.bert_config_file)
transformer.restore_encoder(config.bert_ckpt_file)
learning_rate = CustomSchedule(config.d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

checkpoint = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, config.checkpoint_path, max_to_keep=5)
checkpoint.restore(checkpoint_manager.latest_checkpoint)

_, val_dataset, tokenizer_zh, tokenizer_en, _, val_steps = make_dataset(1)
translator = Translator(tokenizer_zh, tokenizer_en, transformer)

if os.path.exists(config.predict_file):
    with open(config.predict_file, 'r', encoding='utf8') as reader:
        count = len(reader.readlines())
else:
    with open(config.predict_file, 'x', encoding='utf8'):
        count = 0

print('Found {} translated data in {}'.format(count, config.predict_file))

it = iter(val_dataset)
for i in trange(val_steps):
    zh, en = next(it)
    if i < count:
        continue
    pred = translator.beam_search(zh, config.beam_size)
    pred_sentence = tokenizer_en.decode([i for i in pred if i < tokenizer_en.vocab_size])
    with open(config.predict_file, 'a', encoding='utf8') as w:
        w.write(pred_sentence + '\n')
