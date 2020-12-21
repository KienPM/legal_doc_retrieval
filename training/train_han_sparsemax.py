""" Create by Ken at 2020 May 05 """
import os
import argparse
import pickle
import logging
import time
from datetime import datetime
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import tensorflow as tf
from model.han_sparsemax_model import create_model, log_hyper_parameters


def load_training_data():
    training_data = pickle.load(open(train_pickle_file, 'rb'))
    all_queries = []
    all_articles = []
    all_labels = []
    for item in training_data:
        all_queries.append(item[0])
        all_articles.append(item[1])
        all_labels.append(item[2])

    all_queries = np.array(all_queries, dtype='int32')
    all_articles = np.array(all_articles, dtype='int32')
    all_labels = np.array(all_labels, dtype='int32')
    return all_queries, all_articles, all_labels


def train_generator(all_queries, all_articles, all_labels):
    while True:
        num_examples = len(all_queries)
        indices = np.arange(num_examples)
        np.random.shuffle(indices)
        batches = [indices[range(batch_size * i, min(num_examples, batch_size * (i + 1)))] for i in
                   range(num_examples // batch_size + 1)]
        for batch in batches:
            queries = [all_queries[batch]]
            articles = all_articles[batch]
            articles_split = [articles[:, k, :] for k in range(articles.shape[1])]
            labels = all_labels[batch]
            yield queries + articles_split, labels


class TimeCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(TimeCallback, self).__init__()
        self.begin = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.begin = time.time()

    def on_epoch_end(self, epoch, logs=None):
        taken_time = time.time() - self.begin
        logger.info(f'Time taken for epoch {epoch}: {taken_time}s')


def train():
    log_hyper_parameters(logger)
    logger.info(f"batch size: {batch_size}")
    logger.info(f"learning rate: {learning_rate}")

    strategy = tf.distribute.MirroredStrategy()
    logger.info('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model, _, _ = create_model()
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                      metrics=['acc'])

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir='tensorboard/{}'.format(exp_name),
        update_freq=10,
        profile_batch=0
    )

    model_save_path = 'trained_models/' + exp_name
    os.makedirs(model_save_path, exist_ok=True)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        model_save_path + '/' + 'ep{epoch:03d}-loss{loss:.3f}.h5',
        monitor='loss', save_weights_only=True, save_best_only=True, save_freq='epoch')

    time_callback = TimeCallback()

    all_queries, all_articles, all_labels = load_training_data()
    train_gen = train_generator(all_queries, all_articles, all_labels)
    model.fit(train_gen, epochs=epochs, steps_per_epoch=len(all_queries) // batch_size,
              callbacks=[tensorboard_callback, time_callback, checkpoint_callback])


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='HAN model')
    arg_parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of epochs'
    )
    arg_parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size'
    )
    arg_parser.add_argument(
        '--init_lr',
        type=float,
        default=1e-3,
        help='Init learning rate'
    )
    arg_parser.add_argument(
        '--train_pickle_file',
        type=str,
        help='Path to train pickle file'
    )
    arg_parser.add_argument(
        '--exp_name',
        type=str,
        help='Experiment name'
    )
    args = arg_parser.parse_args()

    exp_name = args.exp_name
    os.makedirs('logs', exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logging.basicConfig(filename="logs/{}_train_{}.log".format(exp_name, current_time),
                        format='%(asctime)-15s %(name)s - %(levelname)s %(message)s')
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)

    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.init_lr
    train_pickle_file = args.train_pickle_file

    train()
