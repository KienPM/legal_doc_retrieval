""" Create by Ken at 2020 May 07 """
import os
import argparse
from tqdm import tqdm
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
from pymongo import MongoClient

from model.han_sparsemax_model import create_model

arg_parser = argparse.ArgumentParser(description='Encode articles')
arg_parser.add_argument(
    '--db_host',
    type=str,
    default='localhost',
    help='Mongo DB host'
)
arg_parser.add_argument(
    '--db_port',
    type=int,
    default=5007,
    help='Mongo DB port'
)
arg_parser.add_argument(
    '--db_name',
    type=str,
    default='prd_legal_doc',
    help='DB name'
)
arg_parser.add_argument(
    '--db_input_collection',
    type=str,
    default='text_to_seq_v4',
    help='Input collection name'
)
arg_parser.add_argument(
    '--article_encoder_weights_path',
    type=str,
    help='Path article encoder checkpoint'
)
arg_parser.add_argument(
    '--max_num_sen',
    type=int,
    default=30,
    help='Max number of sentences per article'
)
arg_parser.add_argument(
    '--max_sen_len',
    type=int,
    default=25,
    help='Max sentence len'
)
arg_parser.add_argument(
    '--exp_name',
    type=str,
    help='Experiment name'
)
args = arg_parser.parse_args()
exp_name = args.exp_name
db_host = args.db_host
db_port = args.db_port
db_name = args.db_name
db_input_collection = args.db_input_collection
max_num_sen = args.max_num_sen
max_sen_len = args.max_sen_len


def pad_sentence(seq):
    """
    Padding
    Split if len > max seq len
    """
    res = []
    seq_len = len(seq)
    i = 0
    while i < seq_len:
        temp = seq[i:i + max_sen_len]
        temp_len = len(temp)
        if temp_len < max_sen_len:
            temp.extend([0] * (max_sen_len - temp_len))
        res.append(temp)
        i += max_sen_len

    return res


def pad_article(vec):
    vec = vec[:max_num_sen]
    if len(vec) < max_num_sen:
        vec.extend([[0] * max_sen_len] * (max_num_sen - len(vec)))
    return vec


_, _, article_encoder = create_model()
article_encoder_weights_path = f'trained_models/{exp_name}/article_encoder.h5'
print(f'>>> Load article encoder weights from {article_encoder_weights_path}')
article_encoder.load_weights(article_encoder_weights_path)
print(f'>>> max_num_sen: {max_num_sen}')
print(f'>>> max_sen_len: {max_sen_len}')

mongo_client = MongoClient(db_host, db_port)
db = mongo_client[db_name]
input_collection = db[db_input_collection]
output_collection = db[f'{exp_name}_encoded_articles']
output_collection.delete_many({})

records = list(input_collection.find())
for record in tqdm(records):
    doc_code = record['so_hieu']
    articles = record['cac_dieu']
    for article in articles:
        article_rep = None

        if article["noi_dung"] is not None:
            sentences = article["noi_dung"]
            article_seq = []
            for s in sentences:
                article_seq.extend(pad_sentence(s))
            article_seq = pad_article(article_seq)
            article_seq = np.array(article_seq, dtype='int32')
            article_seq = article_seq[np.newaxis, :]
            article_rep = article_encoder(article_seq)
            article_rep = tf.keras.backend.eval(article_rep)
            article_rep = article_rep[0].tolist()

        output_collection.insert_one({
            'so_hieu_vb': doc_code.lower(),
            'ten_dieu': article['ten_dieu'].lower(),
            'vector': article_rep
        })
