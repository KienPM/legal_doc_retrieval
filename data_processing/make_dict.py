""" Create by Ken at 2020 Mar 06 """
import argparse
from sklearn.feature_extraction.text import CountVectorizer
from pymongo import MongoClient
from tqdm import tqdm
import numpy as np
from utils.string_utils import compound_unicode

arg_parser = argparse.ArgumentParser(description='Make stopwords')
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
    '--collection',
    type=str,
    default='word_tokenized',
    help='Collection name'
)
arg_parser.add_argument(
    '--min_df',
    type=int,
    default=0,
    help='Min DF'
)
arg_parser.add_argument(
    '--threshold',
    type=float,
    default=1.5,
    help='Threshold to decide if a word is stopwords'
)
args = arg_parser.parse_args()

mongo_client = MongoClient(args.db_host, args.db_port)
db = mongo_client[args.db_name]
collection = db[args.collection]


def load_data():
    docs = []
    records = list(collection.find())
    for record in tqdm(records):
        texts = [compound_unicode(record['ten_van_ban'])]
        for clause in record['cac_dieu']:
            texts.append(compound_unicode(clause['tieu_de']))
            texts.append(compound_unicode(clause['noi_dung']))
        docs.append(' '.join(texts))

    return docs


def tokenizer(s):
    return [token for token in s.split(' ') if token != '']


def process(documents):
    """
    :param documents: list of documents
    """
    vectorizer = CountVectorizer(min_df=args.min_df, tokenizer=tokenizer)
    vector = vectorizer.fit_transform(documents)
    vector = vector.toarray()
    vector = np.sum(vector, axis=0)

    dictionary = []
    for term in vectorizer.vocabulary_:
        index = vectorizer.vocabulary_[term]
        dictionary.append({
            'id': index + 1,
            'term': term,
            'occurrences': vector[index]
        })

    dictionary = sorted(dictionary, key=lambda x: x['id'])

    out = open('output/dict.txt', 'w')
    for item in dictionary:
        # term, id, number of occurrences
        out.write('{},{},{}\n'.format(item['id'], item['term'], item['occurrences']))


if __name__ == '__main__':
    docs = load_data()
    process(docs)
    # lines = open('output/dict.txt').readlines()
    # count = 0
    # temp = []
    # for line in lines:
    #     tokens = line.split(',')
    #     f = int(tokens[-1].strip())
    #     if f > 20:
    #         count += 1
    #     else:
    #         temp.append(tokens[1])
    # print(count)
    # print(temp)
