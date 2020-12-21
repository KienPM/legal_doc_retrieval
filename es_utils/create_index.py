# -*- coding: utf-8 -*-
""" Create by Ken at 2020 Feb 06 """
import argparse
import re

from elasticsearch import Elasticsearch
from elasticsearch_dsl import document, Text, Keyword, Date
from elasticsearch_dsl.connections import connections
from tqdm import tqdm
from pymongo import MongoClient

DOCUMENT_INDEX_LEVEL = 'document_level'  # Mức văn bản
ARTICLE_INDEX_LEVEL = 'article_level'  # Mức điều
CLAUSE_INDEX_LEVEL = 'clause_level'  # Mức khoản

arg_parser = argparse.ArgumentParser(description='Create index')

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
    '--db_collection',
    type=str,
    default='word_tokenized',
    help='MongoDB doc_collection name'
)

arg_parser.add_argument(
    '-l',
    '--index_level',
    type=str,
    default="document_level",
    help='Level to create index: {"document_level", "article_level", "clause_level"}'
)

arg_parser.add_argument(
    '-s',
    '--similarity',
    type=str,
    default="tf_idf",
    help='Similarity for scoring: {"BM25", "tf_idf", "boolean"}'
)

args = arg_parser.parse_args()
mongo_client = MongoClient(args.db_host, args.db_port)
db = mongo_client[args.db_name]
collection = db[args.db_collection]
index_level = args.index_level
similarity = args.similarity
index_name = 'index_{}_{}'.format(similarity.lower(), index_level)
print("Index name: ", index_name)
date_regex = re.compile(r'\d{2}/d{2}/d{4}')


class VBPL(document.DocType):
    id = Keyword(similarity="boolean")
    so_hieu = Keyword(similarity="boolean")
    ten_van_ban = Text(similarity=similarity)
    ten_van_ban_raw = Text()
    ngay_ban_hanh = Date()
    ngay_hieu_luc = Date()
    hieu_luc = Keyword(similarity="boolean")
    files = Text()
    link = Text()
    if index_level == ARTICLE_INDEX_LEVEL:
        ten_dieu = Keyword(similarity="boolean")
        tieu_de = Text(similarity=similarity)
        tieu_de_raw = Text()
    if index_level == CLAUSE_INDEX_LEVEL:
        ten_dieu = Keyword(similarity="boolean")
        ten_khoan = Keyword(similarity="boolean")
        tieu_de = Text(similarity=similarity)
        tieu_de_raw = Text()
        noi_dung_raw = Text()
    noi_dung = Text(similarity=similarity)

    class Meta:
        """
        Phải khai báo lại class Meta với doc_type và tên index đúng như dưới
        """
        doc_type = 'vbpl'
        index = index_name

    class Index:
        """
        Từ bản 0.10.2 trở đi phải khai báo thêm cả class Index có thuộc tính name là tên index trong elastic search như dưới đây
        """
        name = index_name


# Tạo kết nối
es = Elasticsearch()
connections.create_connection(hosts=['localhost'], timeout=20)
connections.add_connection('vbpl', es)  # Thêm một doc_type có tên vbpl vào connections
VBPL.init(index_name)  # Khởi tạo class Doctype với tên Index


def insert(doc):
    id_ = doc["id"]
    so_hieu = doc["so_hieu"].lower()
    ten_van_ban = doc["ten_van_ban"].lower()
    ten_van_ban_raw = doc["ten_van_ban_raw"]
    ngay_ban_hanh = doc["ngay_ban_hanh"]
    if 'ngay_hieu_luc' in doc:
        ngay_hieu_luc = doc["ngay_hieu_luc"]
    else:
        ngay_hieu_luc = ''
    if 'hieu_luc' in doc:
        hieu_luc = doc["hieu_luc"].lower()
    else:
        hieu_luc = ''
    if "file" in doc:
        files = '\n'.join(doc["file"])
    else:
        files = []
    link = doc["link"]
    if index_level == DOCUMENT_INDEX_LEVEL:
        vbpl = VBPL(id=id_, so_hieu=so_hieu, ten_van_ban=ten_van_ban, ten_van_ban_raw=ten_van_ban_raw,
                    hieu_luc=hieu_luc, files=files, link=link)
        if date_regex.match(ngay_ban_hanh):
            vbpl.ngay_ban_hanh = ngay_ban_hanh
        if date_regex.match(ngay_hieu_luc):
            vbpl.ngay_hieu_luc = ngay_hieu_luc
        clause_content = [clause["tieu_de"] + ' ' + clause["noi_dung"] for clause in doc["cac_dieu"]]
        vbpl.noi_dung = ' '.join(clause_content)
        vbpl.save()
    elif index_level == ARTICLE_INDEX_LEVEL:
        for clause in doc['cac_dieu']:
            vbpl = VBPL(id=id_, so_hieu=so_hieu, ten_van_ban=ten_van_ban, ten_van_ban_raw=ten_van_ban_raw,
                        hieu_luc=hieu_luc, files=files, link=link)
            if date_regex.match(ngay_ban_hanh):
                vbpl.ngay_ban_hanh = ngay_ban_hanh
            if date_regex.match(ngay_hieu_luc):
                vbpl.ngay_hieu_luc = ngay_hieu_luc
            vbpl.ten_dieu = clause["ten_dieu"].lower()
            vbpl.tieu_de = clause["tieu_de"]
            vbpl.tieu_de_raw = clause["tieu_de_raw"]
            vbpl.noi_dung = clause["tieu_de"] + ' ' + clause["noi_dung"]
            vbpl.noi_dung_raw = clause["noi_dung_raw"]
            vbpl.save()


if __name__ == '__main__':
    docs = list(collection.find())
    for d in tqdm(docs):
        insert(d)
