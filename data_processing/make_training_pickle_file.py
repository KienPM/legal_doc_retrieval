"""
Create by Ken at 2020 Apr 30
Create pickle file for training
Data structure for each example: [query sequence, articles, labels corresponding to articles (1~positive, 0~negative)]
v4
"""
import os
import argparse

import numpy as np
import random
import pickle

from tqdm import tqdm
from pymongo import MongoClient


def pad_query(vec):
    vec = vec[:max_query_len]
    if len(vec) < max_query_len:
        vec.extend([0] * (max_query_len - len(vec)))
    return vec


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


def load_all_articles():
    all_articles = []
    dict_ = {}
    docs = text_to_seq_collection.find()
    for doc in docs:
        doc_code = doc['so_hieu'].lower()
        for article in doc['cac_dieu']:
            article_name = article['ten_dieu'].lower()
            dict_[f'{article_name}@{doc_code}'] = article['noi_dung']
            if article['noi_dung'] is not None:
                all_articles.append(f'{article_name}@{doc_code}')
    return all_articles, dict_


def main():
    all_articles, all_articles_map = load_all_articles()
    examples = []
    records = list(training_data_collection.find())
    for record in tqdm(records):
        query = pad_query(record["query"])
        record_negative = record["negative"][:num_es_negative + 5]
        taken = set(record["positive"] + record_negative)
        num_taken = len(taken)

        for positive in record["positive"]:
            if random_es_negative and len(record_negative) > num_es_negative:
                es_negative = random.sample(record_negative, num_es_negative)
            else:
                es_negative = record_negative[:num_es_negative]

            temp_num_random_negative = num_random_negative + (num_es_negative - len(es_negative))
            random_negative = random.sample(all_articles, temp_num_random_negative + num_taken * 2)
            for neg in random_negative:
                if all_articles_map[neg] is None:
                    random_negative.remove(neg)
            random_negative = list(set(random_negative) - taken)[:temp_num_random_negative]

            negative = es_negative + random_negative
            len_negative = len(negative)
            assert len_negative == num_es_negative + num_random_negative, 'Not enough random negative'

            articles = [positive] + negative
            articles_seq = []
            for a in articles:
                content = all_articles_map[a]
                a_seq = []
                for s in content:
                    a_seq.extend(pad_sentence(s))
                a_seq = pad_article(a_seq)
                articles_seq.append(a_seq)

            labels = [0] * len(articles)
            labels[0] = 1

            ids = np.arange(len(articles))
            np.random.shuffle(ids)
            examples.append([
                np.asarray(query),
                np.asarray([articles_seq[i] for i in ids]),
                np.asarray([labels[i] for i in ids])
            ])

    parent_dir = os.path.dirname(output_file)
    os.makedirs(parent_dir, exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(examples, f)
        f.close()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Make pickle file for training')
    arg_parser.add_argument(
        '--db_host',
        type=str,
        default='localhost',
        help='MongoDB host'
    )
    arg_parser.add_argument(
        '--db_port',
        type=int,
        default=5007,
        help='MongoDB port'
    )
    arg_parser.add_argument(
        '--db_name',
        type=str,
        default='prd_legal_doc',
        help='MongoDB input DB name'
    )
    arg_parser.add_argument(
        '--db_training_data_collection',
        type=str,
        default='training_data',
        help='MongoDB training data collection name'
    )
    arg_parser.add_argument(
        '--db_text_to_seq_collection',
        type=str,
        default='text_to_seq_v4',
        help='MongoDB text to seq collection name'
    )
    arg_parser.add_argument(
        '--num_es_negative',
        type=int,
        default=50,
        help='Number of ES negative articles per example'
    )
    arg_parser.add_argument(
        '--num_random_negative',
        type=int,
        default=70,
        help='Number of negative articles per example'
    )
    arg_parser.add_argument(
        '--max_query_len',
        type=int,
        default=40,
        help='Max query len'
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
        '--random_es_negative',
        type=bool,
        default=True,
        help='Random sample ES negative'
    )
    arg_parser.add_argument(
        '--output_file',
        type=str,
        default='text_to_seq_v4',
        help='MongoDB text to seq collection name'
    )
    args = arg_parser.parse_args()

    mongo_client = MongoClient(args.db_host, args.db_port)
    db = mongo_client[args.db_name]
    training_data_collection = db[args.db_training_data_collection]
    text_to_seq_collection = db[args.db_text_to_seq_collection]
    num_es_negative = args.num_es_negative
    num_random_negative = args.num_random_negative
    max_query_len = args.max_query_len
    max_num_sen = args.max_num_sen
    max_sen_len = args.max_sen_len
    random_es_negative = args.random_es_negative
    output_file = args.output_file
    print('> max_query_len: ', max_query_len)
    print('> max_num_sen: ', max_num_sen)
    print('> max_sen_len: ', max_sen_len)
    print('> random_es_negative: ', random_es_negative)
    print('> num_es_negative: ', num_es_negative)
    print('> num_random_negative: ', num_random_negative)
    print('> training data collection: ', args.db_training_data_collection)

    main()
