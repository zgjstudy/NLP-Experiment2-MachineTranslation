import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

import time
import logging
import tensorflow as tf

import config
from prepare_dataset import make_dataset
from transformer import Transformer
from utils import CustomSchedule, create_masks

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


mirrored_strategy = tf.distribute.MirroredStrategy()
print('Number of devices for distributed training: {}'.format(mirrored_strategy.num_replicas_in_sync))
GLOBAL_BATCH_SIZE = config.batch_size_per_replica * mirrored_strategy.num_replicas_in_sync

train_dataset, val_dataset, tokenizer_zh, tokenizer_en, train_steps_per_epoch, _ = make_dataset(GLOBAL_BATCH_SIZE)

with mirrored_strategy.scope():
    transformer = Transformer(config.dec_config, config.bert_config_file)
    transformer.restore_encoder(config.bert_ckpt_file)
    learning_rate = CustomSchedule(config.d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    checkpoint = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, config.checkpoint_path, max_to_keep=5)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                reduction=tf.keras.losses.Reduction.NONE)

    def loss_function(target, pred):
        mask = tf.math.logical_not(tf.math.equal(target, 0))
        loss_ = loss_object(target, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.nn.compute_average_loss(loss_, global_batch_size=GLOBAL_BATCH_SIZE)

    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

dist_dataset_train = mirrored_strategy.experimental_distribute_dataset(train_dataset)


if checkpoint_manager.latest_checkpoint:
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    print('Latest checkpoint restored from ' + config.checkpoint_path)

file_path = './log/'
# 创建一个logger
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)

# 创建一个handler，
timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
fh = logging.FileHandler(file_path + 'gigaword_log_' + timestamp + '.txt')
fh.setLevel(logging.DEBUG)

# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# 定义handler的输出格式
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# 给logger添加handler
logger.addHandler(fh)
logger.addHandler(ch)

with mirrored_strategy.scope():
    def train_step(inputs):
        src_inputs, target = inputs
        target_inputs = target[:, :-1]
        target_real = target[:, 1:]

        target_mask, src_mask = create_masks(src_inputs, target_inputs)

        with tf.GradientTape() as tape:
            predictions = transformer(src_inputs, target_inputs, target_mask, src_mask, True)
            loss = loss_function(target_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_accuracy.update_state(target_real, predictions)
        return loss

    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = mirrored_strategy.experimental_run_v2(train_step, args=(dataset_inputs,))
        return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


def train():
    logger.info('Start training')
    with mirrored_strategy.scope():
        for epoch in range(0, config.epochs):
            epoch_start = time.time()
            batch100_start = time.time()
            total_loss = 0.0
            num_batches = 0
            for x in dist_dataset_train:
                batch_loss = distributed_train_step(x)
                total_loss += batch_loss
                num_batches += 1
                if num_batches % 100 == 0:
                    template = 'Epoch {}  Batch {}  Loss {:.6f}  {:.2f} sec'
                    logger.info(template.format(epoch + 1, num_batches, batch_loss, time.time() - batch100_start))
                    batch100_start = time.time()

            logger.info('Epoch {} Loss {:.6f} Accuracy {}'.format(epoch + 1,
                                                                  total_loss / train_steps_per_epoch,
                                                                  train_accuracy.result()*100))
            logger.info('Time taken for 1 epoch {:.2f} sec\n'.format(time.time() - epoch_start))
            checkpoint_manager.save()
            train_accuracy.reset_states()


if __name__ == '__main__':
    train()
