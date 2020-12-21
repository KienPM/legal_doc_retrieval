""" Create by Ken at 2020 Mar 12 """
import argparse
import random
from pymongo import MongoClient

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
    default=27017,
    help='MongoDB port'
)
arg_parser.add_argument(
    '--db_name',
    type=str,
    default='legal_doc',
    help='MongoDB DB name'
)
arg_parser.add_argument(
    '--db_ground_truth_collection',
    type=str,
    default='ground_truth',
    help='MongoDB ground truth collection name'
)
arg_parser.add_argument(
    '--db_train_ground_truth_collection',
    type=str,
    default='train_ground_truth',
    help='MongoDB positive ground truth collection name'
)
arg_parser.add_argument(
    '--db_test_ground_truth_collection',
    type=str,
    default='test_ground_truth',
    help='MongoDB negative ground truth collection name'
)
arg_parser.add_argument(
    '--train_ratio',
    type=float,
    default=0.7,
    help='MongoDB negative ground truth collection name'
)
args = arg_parser.parse_args()

mongo_client = MongoClient(args.db_host, args.db_port)
db = mongo_client[args.db_name]
gt_collection = db[args.db_ground_truth_collection]
train_gt_collection = db[args.db_train_ground_truth_collection]
test_gt_collection = db[args.db_test_ground_truth_collection]
train_ratio = args.train_ratio

print("Clean training and testing ground truth collections__")
train_gt_collection.remove()
test_gt_collection.remove()

print("Fetching...")
records = list(gt_collection.find())

print("Splitting...")
num = len(records)
random.shuffle(records)
num_train = int(num * train_ratio)
train_records = records[:num_train]
test_records = records[num_train:]

print("Writing train records...")
train_gt_collection.insert_many(train_records)

print("Writing test records...")
test_gt_collection.insert_many(test_records)
