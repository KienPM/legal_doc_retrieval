"""
Create by Ken at 2020 Apr 30
Convert text to int sequence
v4
"""
import os
import re
import argparse
import string
import logging
from datetime import datetime

from pymongo import MongoClient
from tqdm import tqdm
from underthesea import sent_tokenize
from CocCocTokenizer import PyTokenizer
from html import unescape

from utils.string_utils import compound_unicode, fix_font

os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename="logs/text_to_seq_runtime_{}.log".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S")),
                    format='%(asctime)-15s %(name)s - %(levelname)s %(message)s')
logger = logging.getLogger("text_to_seq")
logger.setLevel(logging.WARNING)

arg_parser = argparse.ArgumentParser(description='Text to sequence')
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
    help='MongoDB DB name'
)
arg_parser.add_argument(
    '--db_input_collection',
    type=str,
    default='original',
    help='MongoDB input collection name'
)
arg_parser.add_argument(
    '--db_output_collection',
    type=str,
    default='text_to_seq_v4',
    help='MongoDB output collection name'
)
arg_parser.add_argument(
    '--dict_file',
    type=str,
    default='output/prd_dict.txt',
    help='Path to dict file'
)
arg_parser.add_argument(
    '--max_sentence_len',
    type=int,
    default=40,
    help='Max sentence length'
)


def parse_dict_file():
    lines = open(args.dict_file, 'r').readlines()
    dict_ = {}
    for line in lines:
        tokens = line.split(',')
        id_ = int(tokens[0].strip())
        term = tokens[1].strip()
        dict_[term] = id_
    return dict_


def pre_process_text(s):
    """
    - Tokenize words using coccoc-tokenizer
    - Eliminate punctuation
    - Transform to lower case
    :type s: str
    :rtype: str
    """
    s = compound_unicode(s)
    s = ' '.join(tokenizer.word_tokenize(s, tokenize_option=2))
    s = unescape(s)
    s = s.translate(translator)
    s = s.lower()
    s = fix_font(s)
    return s


def process_text(s):
    """
    Convert text to int sequence
    :type s: str
    :rtype: list[int]
    """
    seq = []
    s = pre_process_text(s)
    if s == '':
        return None

    terms = s.split(' ')
    for term in terms:
        if term in text_to_seq_dict:
            seq.append(text_to_seq_dict[term])
        else:
            seq.append(0)

    return seq


def process_article(content):
    """
    Convert text to int sequence
    :type content: str
    :rtype: list[list[int]]
    """
    lines = content.split('\n')
    res = []
    for line in lines:
        sentences = sent_tokenize(line)
        for s in sentences:
            s = re.sub(r'^\d+\s*[.:)/\-]|^[a-z]\s*[.:)/\-]|^[MDCLXVI]+\s*[.:)/\-]', '', s)
            seq = process_text(s)
            if seq is not None:
                res.append(seq)

    return res


def process_doc(document):
    seq_doc = {
        "id": document["id"],
        "so_hieu": document["so_hieu"],
        "ten_van_ban": document["ten_van_ban"],
        "cac_dieu": document["cac_dieu"]
    }
    for clause in seq_doc["cac_dieu"]:
        if clause["tieu_de"] == '#0' or clause["tieu_de"] == '':
            content = clause["noi_dung"]
        else:
            content = clause["tieu_de"] + '\n' + clause["noi_dung"]
        clause["noi_dung"] = process_article(content)

    output_collection.insert_one(seq_doc)


if __name__ == '__main__':
    args = arg_parser.parse_args()
    mongo_client = MongoClient(args.db_host, args.db_port)
    db = mongo_client[args.db_name]
    input_collection = db[args.db_input_collection]
    output_collection = db[args.db_output_collection]
    output_collection.delete_many({})
    max_sentence_len = args.max_sentence_len

    src_dict = string.punctuation.replace('_', '\n')
    dst_dict = ' ' * len(src_dict)
    translator = str.maketrans(src_dict, dst_dict)
    tokenizer = PyTokenizer(load_nontone_data=True)
    text_to_seq_dict = parse_dict_file()

    docs = list(input_collection.find())
    for doc in tqdm(docs):
        process_doc(doc)
