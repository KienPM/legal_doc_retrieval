"""
Create by Ken at 2020 Feb 17
- Transform to lower case
- Tokenize words
- Eliminate punctuation
"""
import argparse
import string

from pymongo import MongoClient
from tqdm import tqdm
from CocCocTokenizer import PyTokenizer
from html import unescape
from string_utils import fix_font

arg_parser = argparse.ArgumentParser(description='Word tokenizing')
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
    default='legal_doc',
    help='MongoDB DB name'
)
arg_parser.add_argument(
    '--db_input_collection',
    type=str,
    default='original',
    help='MongoDB input doc_collection name'
)
arg_parser.add_argument(
    '--db_output_collection',
    type=str,
    default='word_tokenized',
    help='MongoDB output doc_collection name'
)
args = arg_parser.parse_args()

mongo_client = MongoClient(args.db_host, args.db_port)
db = mongo_client[args.db_name]
input_collection = db[args.db_input_collection]
output_collection = db[args.db_output_collection]
src_dict = string.punctuation.replace('_', '\n')
dst_dict = ' ' * len(src_dict)
translator = str.maketrans(src_dict, dst_dict)
tokenizer = PyTokenizer(load_nontone_data=True)


def process_string(s):
    """
    - Tokenize words using coccoc-tokenizer
    - Eliminate punctuation
    - Transform to lower case
    :type s: str
    :rtype: str
    """
    # s = compound_unicode(s)
    s = ' '.join(tokenizer.word_tokenize(s, tokenize_option=2))
    s = unescape(s)
    s = s.translate(translator)
    s = s.lower()
    s = fix_font(s)
    return s


def process_doc(document):
    document["ten_van_ban_raw"] = document["ten_van_ban"]
    document["ten_van_ban"] = process_string(document["ten_van_ban"])
    for clause in document["cac_dieu"]:
        clause["tieu_de_raw"] = clause["tieu_de"]
        clause["tieu_de"] = process_string(clause["tieu_de"])
        clause["noi_dung_raw"] = clause["noi_dung"]
        clause["noi_dung"] = process_string(clause["noi_dung"])
    output_collection.insert_one(document)


if __name__ == '__main__':
    docs = list(input_collection.find())
    for doc in tqdm(docs):
        process_doc(doc)
