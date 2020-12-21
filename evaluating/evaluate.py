""" Create by Ken at 2020 May 02 """
import os
import argparse
import numpy as np
import time

from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from pymongo import MongoClient

from model.han_sparsemax_model import create_model
from utils.string_utils import process_text
from evaluating.ndcg import n_dcg

tf.get_logger().setLevel('ERROR')

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
    default='legal_doc',
    help='DB name'
)
arg_parser.add_argument(
    '--exp_db_name',
    type=str,
    default='exp_legal_doc',
    help='MongoDB output DB name'
)
arg_parser.add_argument(
    '--db_test_collection',
    type=str,
    default='test_ground_truth',
    help='Test ground truth collection name'
)
arg_parser.add_argument(
    '--db_encoded_article_collection',
    type=str,
    default='encoded_articles',
    help='Test encoded article collection name'
)
arg_parser.add_argument(
    '--es_output_file',
    type=str,
    default='data/bm25_article_recall_5000_unlimited_output.csv',
    help='Path to ES output file'
)
arg_parser.add_argument(
    '--es_output_limit',
    type=int,
    help='Limit number of results from ES output file'
)
arg_parser.add_argument(
    '--exp_name',
    type=str,
    help='Experiment name'
)
arg_parser.add_argument(
    '--dict_file',
    type=str,
    default='data/dict.txt',
    help='Path to dict file'
)


def pad_query(vec, max_query_len):
    vec = vec[:max_query_len]
    if len(vec) < max_query_len:
        vec.extend([0] * (max_query_len - len(vec)))
    return vec


def parse_dict_file():
    lines = open(args.dict_file, 'r').readlines()
    dict_ = {}
    for line in lines:
        tokens = line.split(',')
        id_ = int(tokens[0].strip())
        term = tokens[1].strip()
        dict_[term] = id_
    return dict_


def parse_experiment_file():
    experiment_data = {}
    lines = open(es_output_file, 'r').readlines()[:-1]
    for line in lines:
        parts = line.strip().split(',')
        query_id = parts[0]
        exp_out = parts[2].strip().split('|')[:es_output_limit]
        experiment_data[query_id] = exp_out
    return experiment_data


def run():
    _, query_encoder, _ = create_model()
    print('Loading query encider weights from {}...'.format(query_encoder_weights_path))
    query_encoder.load_weights(query_encoder_weights_path)

    mongo_client = MongoClient(db_host, db_port)
    exp_db = mongo_client[args.exp_db_name]
    test_collection = exp_db[db_test_collection]
    encoded_article_collection = exp_db[db_encoded_article_collection]

    encoded_articles = {}
    print('Loading encoded articles from {}...'.format(db_encoded_article_collection))
    for record in tqdm(list(encoded_article_collection.find())):
        doc_code = record['so_hieu_vb'].lower()
        article_name = record['ten_dieu'].lower()
        encoded_articles[f'{article_name}@{doc_code}'] = record['vector']

    examples = list(test_collection.find())
    os.makedirs('output', exist_ok=True)
    output = open('output/{}_es_limit_{}_recall_20.csv'.format(exp_name, es_output_limit), 'w')
    total_recall = 0
    total_n_dcg = 0
    ignore_examples = 0  # Number of examples don't contain article ground truth

    start_time = time.time()
    for example in tqdm(examples):
        ground_truth = []
        for document in example["documents"]:
            doc_code = document['code'].lower()
            if 'articles' in document:
                for article in document['articles']:
                    article_name = article['name'].lower()
                    ground_truth.append('{}@{}'.format(article_name, doc_code))

        es_res = es_output[str(example['_id'])]
        articles = []
        articles_rep = []
        for item in es_res:
            if item not in encoded_articles:
                print(f'cannot found {item}')
                continue
            article = encoded_articles[item]
            if article is not None:
                articles.append(item)
                articles_rep.append(article)
        articles_rep = np.array(articles_rep)

        query = process_text(example["query"], text_to_seq_dict)
        query = pad_query(query, 40)
        query = np.array(query, dtype='int32')
        query = query[np.newaxis, :]
        query_rep = query_encoder(query)

        group_size = len(articles_rep)
        query_rep = tf.tile(query_rep, [group_size, 1])
        scores = tf.keras.layers.dot([query_rep, articles_rep], axes=-1)
        scores = tf.reshape(scores, (group_size,))

        scores = tf.keras.backend.eval(scores)
        articles_scores = []
        for i in range(len(articles)):
            articles_scores.append({
                'article': articles[i],
                'score': scores[i].item()
            })

        articles_scores = sorted(articles_scores, key=lambda x: x['score'], reverse=True)
        predicted_articles = [article_score['article'] for article_score in articles_scores[:20]]
        scores = [str(article_score['score']) for article_score in articles_scores[:20]]
        intersection = set(ground_truth) & set(predicted_articles)
        if len(ground_truth) == 0:
            recall = 0
            n_dcg_score = 0
            ignore_examples += 1
        else:
            recall = len(intersection) / len(ground_truth)
            n_dcg_score = n_dcg(predicted_articles, ground_truth)

        total_recall += recall
        total_n_dcg += n_dcg_score
        output.write("{},{},{},{},{}\n".format(example["_id"], recall, n_dcg_score, '|'.join(predicted_articles),
                                               ','.join(scores)))

    print('time: ', time.time() - start_time)
    output.write("Average recall: {},Average nDCG: {}".format(
        total_recall / (len(examples) - ignore_examples),
        total_n_dcg / (len(examples) - ignore_examples))
    )
    output.close()


if __name__ == '__main__':
    args = arg_parser.parse_args()
    exp_name = args.exp_name
    db_host = args.db_host
    db_port = args.db_port
    db_name = args.db_name
    db_test_collection = exp_name + '_' + args.db_test_collection
    db_encoded_article_collection = exp_name + '_' + args.db_encoded_article_collection
    query_encoder_weights_path = "trained_models/{}/query_encoder.h5".format(exp_name)
    es_output_file = args.es_output_file
    es_output_limit = args.es_output_limit
    print(f'>>> es_output_file: {es_output_file}')
    print(f'>>> es_output_limit: {es_output_limit}')
    print(f'>>> Dict file: {args.dict_file}')
    print(f'>>> query_encoder_weights_path: {query_encoder_weights_path}')

    text_to_seq_dict = parse_dict_file()
    es_output = parse_experiment_file()
    run()
