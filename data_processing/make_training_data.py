"""
Create by Ken at 2020 Jun 17
Make training data
v4
"""
import os
import argparse
from datetime import datetime
import logging

from tqdm import tqdm
from pymongo import MongoClient

from utils.string_utils import process_text


def load_all_articles():
    all_articles = []
    dict_ = {}
    docs = text_to_sqe_collection.find()
    for doc in docs:
        doc_code = doc['so_hieu'].lower()
        for article in doc['cac_dieu']:
            article_name = article['ten_dieu'].lower()
            dict_[f'{article_name}@{doc_code}'] = article['noi_dung']
            if article['noi_dung'] is not None:
                all_articles.append(f'{article_name}@{doc_code}')
    return all_articles, dict_


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
    lines = open(args.experiment_file, 'r').readlines()[:-1]
    dict_ = {}
    for line in lines:
        parts = line.split(',')
        query_id = parts[0]
        exp_out = parts[2].split('|')
        dict_[query_id] = exp_out
    return dict_


def main():
    records = list(train_gt_collection.find())
    _, article_dict = load_all_articles()
    for record in tqdm(records):
        _id = record["_id"]
        query = process_text(record["query"], text_to_sqe_dict)
        positive = []
        negative = []

        for gt_doc in record["documents"]:
            if "articles" in gt_doc:
                gt_doc_code = gt_doc["code"]
                for gt_article in gt_doc["articles"]:
                    item = f'{gt_article["name"].lower()}@{gt_doc_code.lower()}'
                    if item in article_dict and article_dict[item] is not None:
                        positive.append(item)
                    else:
                        print("Cannot found article with name: '{}' in document '{}'".format(
                            gt_article["name"], gt_doc["code"])
                        )
                        logger.warning("Cannot found article with name: '{}' in document '{}'".format(
                            gt_article["name"], gt_doc["code"])
                        )

        negative_codes = [item for item in experiment_data[str(_id)] if item not in positive]
        count = 0
        for neg in negative_codes:
            article_name, doc_code = neg.split('@')
            article_name = article_name.title()

            if neg in article_dict and article_dict[neg] is not None:
                negative.append(neg)
                count += 1
                if count == num_negative:
                    break
            else:
                logger.warning("Cannot found article with name: '{}' in document '{}'".format(
                    article_name, doc_code)
                )

        output_collection.insert_one({
            "_id": _id,
            "query": query,
            "positive": positive,
            "negative": negative
        })


if __name__ == '__main__':
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(filename="logs/make_train_data_{}.log".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S")),
                        format='%(asctime)-15s %(name)s - %(levelname)s %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARNING)

    arg_parser = argparse.ArgumentParser(description='Make positive examples')
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
        '--db_train_gt_collection',
        type=str,
        default='ground_truth',
        help='MongoDB train ground truth collection name'
    )
    arg_parser.add_argument(
        '--db_text_to_sqe_collection',
        type=str,
        default='text_to_seq_v4',
        help='MongoDB ground truth collection name'
    )
    arg_parser.add_argument(
        '--experiment_file',
        type=str,
        default='output/prd_bm25_article_recall_1000_limit_5.csv',
        help='Path to experiment file (for getting negative examples)'
    )
    arg_parser.add_argument(
        '--dict_file',
        type=str,
        default='output/prd_dict.txt',
        help='Path to dict file'
    )
    arg_parser.add_argument(
        '--db_output_collection',
        type=str,
        default='training_data',
        help='MongoDB ground truth collection name'
    )
    arg_parser.add_argument(
        '--num_negative',
        type=int,
        default=100,
        help='Maximum number of negative ES results'
    )
    args = arg_parser.parse_args()

    num_negative = args.num_negative
    mongo_client = MongoClient(args.db_host, args.db_port)
    db = mongo_client[args.db_name]
    train_gt_collection = db[args.db_train_gt_collection]
    text_to_sqe_collection = db[args.db_text_to_sqe_collection]
    output_collection = db[args.db_output_collection]
    output_collection.delete_many({})

    experiment_data = parse_experiment_file()
    text_to_sqe_dict = parse_dict_file()
    main()
