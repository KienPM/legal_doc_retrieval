"""
Create by Ken at 2020 June 14
Split each article into sentences
Encode sentences -> encode article
"""
import tensorflow as tf
from tensorflow.keras.layers import *
from model.sparsemax import sparsemax

INPUT_VOCAB_SIZE = 31450
MAX_QUERY_LEN = 40
MAX_NUM_SENTENCES = 30
MAX_SENTENCE_LEN = 25
GROUP_SIZE = 121
D_EMBEDDING = 512
D_CNN = 512
KERNEL_SIZE = 3
D_Q = 200
DROPOUT_RATE = 0.2

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def log_hyper_parameters(logger):
    logger.info("-" * 10)
    logger.info("Hyper parameters")
    logger.info(f"MAX_QUERY_LEN: {MAX_QUERY_LEN}")
    logger.info(f"MAX_NUM_SENTENCES: {MAX_NUM_SENTENCES}")
    logger.info(f"MAX_SENTENCE_LEN: {MAX_SENTENCE_LEN}")
    logger.info(f"INPUT_VOCAB_SIZE: {INPUT_VOCAB_SIZE}")
    logger.info(f"GROUP_SIZE: {GROUP_SIZE}")
    logger.info(f"D_EMBEDDING: {D_EMBEDDING}")
    logger.info(f"D_CNN: {D_CNN}")
    logger.info(f"KERNEL_SIZE: {KERNEL_SIZE}")
    logger.info(f"D_Q: {D_Q}")
    logger.info("-" * 10)


def get_actual_sen_len(x):
    not_zero_mask = tf.cast(tf.math.not_equal(x, 0), tf.float32)
    actual_len = tf.reduce_sum(not_zero_mask, axis=-1)
    return not_zero_mask, actual_len


def create_empty_sen_mask(x):
    sum_sen = tf.reduce_sum(x, axis=-1)
    empty_sen_mask = tf.cast(tf.math.equal(sum_sen, 0), tf.float32)
    return empty_sen_mask


def create_model(embedding_matrix=None):
    """
    Return: Model, Query encoder, Article encoder
    """
    if embedding_matrix is not None:
        embedding_layer = Embedding(INPUT_VOCAB_SIZE, D_EMBEDDING, weights=[embedding_matrix], trainable=True)
    else:
        embedding_layer = Embedding(INPUT_VOCAB_SIZE, D_EMBEDDING, trainable=True)

    # Query branch
    query_input = Input((MAX_QUERY_LEN,), dtype='int32')

    embedded_sequences_query = embedding_layer(query_input)
    embedded_sequences_query = Dropout(DROPOUT_RATE)(embedded_sequences_query)

    query_cnn = Convolution1D(filters=D_CNN, kernel_size=KERNEL_SIZE, padding='same', activation='relu', strides=1)(
        embedded_sequences_query
    )
    query_cnn = Dropout(DROPOUT_RATE)(query_cnn)
    query_cnn = LayerNormalization(epsilon=1e-6)(query_cnn)

    attention_query = Dense(D_Q, activation='tanh')(query_cnn)
    attention_query = Flatten()(Dense(1)(attention_query))
    attention_weight_query = Activation('softmax')(attention_query)
    query_rep = Dot((1, 1))([query_cnn, attention_weight_query])

    query_encoder = tf.keras.Model(query_input, query_rep)

    # Article branch
    # # Sentence encoder
    sentence_input = Input(shape=(MAX_SENTENCE_LEN,), dtype='int32')

    sentence_embedded_sequences = embedding_layer(sentence_input)
    sentence_embedded_sequences = Dropout(DROPOUT_RATE)(sentence_embedded_sequences)

    sentence_cnn = Convolution1D(filters=D_CNN, kernel_size=KERNEL_SIZE, padding='same', activation='relu', strides=1)(
        sentence_embedded_sequences
    )
    sentence_cnn = Dropout(DROPOUT_RATE)(sentence_cnn)
    sentence_cnn = LayerNormalization(epsilon=1e-6)(sentence_cnn)

    sentence_attention = Dense(D_Q, activation='tanh')(sentence_cnn)
    sentence_attention = Flatten()(Dense(1)(sentence_attention))
    sentence_attention_weight = Activation('softmax')(sentence_attention)
    sentence_rep = Dot((1, 1))([sentence_cnn, sentence_attention_weight])

    sentence_rep_attention_w = concatenate([sentence_rep, sentence_attention_weight], axis=1)
    sentence_encoder = tf.keras.Model(sentence_input, sentence_rep_attention_w)

    # # Article encoder
    article_input = Input((MAX_NUM_SENTENCES, MAX_SENTENCE_LEN), dtype='int32')
    sentences_rep_attention_w = TimeDistributed(sentence_encoder)(article_input)
    sentences_rep, sentences_attention_w = tf.split(sentences_rep_attention_w, [D_CNN, MAX_SENTENCE_LEN], axis=-1)

    not_zero_mask, actual_sen_len = get_actual_sen_len(article_input)
    sentences_attention_w = multiply([sentences_attention_w, not_zero_mask])
    article_attention = tf.keras.backend.sum(sentences_attention_w, axis=-1)
    empty_sen_mask = create_empty_sen_mask(article_input)
    article_attention /= (actual_sen_len + empty_sen_mask)  # Add 1 to empty sen to avoid divide by zero
    article_attention_weight = Activation(activation=sparsemax)(article_attention)
    article_rep = Dot((1, 1))([sentences_rep, article_attention_weight])

    article_encoder = tf.keras.Model(article_input, article_rep)

    # Scoring
    articles_input = [Input((MAX_NUM_SENTENCES, MAX_SENTENCE_LEN), dtype='int32') for _ in range(GROUP_SIZE)]
    articles_rep = [article_encoder(articles_input[_]) for _ in range(GROUP_SIZE)]
    logits = [dot([query_rep, a_rep], axes=-1) for a_rep in articles_rep]
    logits = concatenate(logits)
    logits = Activation(tf.keras.activations.softmax)(logits)

    model = tf.keras.Model([query_input] + articles_input, logits)

    return model, query_encoder, article_encoder


if __name__ == '__main__':
    create_model()
