""" Create by Ken at 2020 Mar 28 """
import os

from model.han_sparsemax_model import create_model

model_dir = 'trained_models/exp28'
model_file = 'ep001-loss3.528.ckpt'


def save_models():
    model, query_encoder, article_encoder = create_model()
    model.load_weights(os.path.join(model_dir, model_file))
    query_encoder.save_weights(os.path.join(model_dir, 'query_encoder.h5'))
    article_encoder.save_weights(os.path.join(model_dir, 'article_encoder.h5'))


def save_from_checkpoint():
    model, query_encoder, article_encoder = create_model()
    # latest = tf.train.latest_checkpoint(model_dir)
    # model.load_weights(latest)

    model.load_weights(os.path.join(model_dir, model_file))

    query_encoder.save_weights(os.path.join(model_dir, 'query_encoder.h5'))
    article_encoder.save_weights(os.path.join(model_dir, 'article_encoder.h5'))


if __name__ == '__main__':
    # save_models()
    save_from_checkpoint()
