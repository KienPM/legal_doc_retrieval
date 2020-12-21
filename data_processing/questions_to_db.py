""" Create by Ken at 2020 Feb 05 """
import os
import re
import json
from glob import glob
from pymongo import MongoClient

QUESTION_LEVEL = 0  # Câu hỏi
QUERY_LEVEL = 1  # Các cách hỏi khác nhau
DOCUMENT_LEVEL = 2  # Văn bản
ARTICLE_LEVEL = 3  # Điều
CLAUSE_LEVEL = 4  # Khoản
POINT_LEVEL = 5  # Điểm

mongo_client = MongoClient('localhost', 27017)
db = mongo_client['legal_doc']
collection = db['ground_truth']

question_re = re.compile(r'^\d+\.\s+\d+\n$')


def check_level(line):
    """
    :type line: str
    """
    line = line.strip()

    if question_re.match(line):
        return QUESTION_LEVEL

    if line.startswith('-'):
        return QUERY_LEVEL

    if line.startswith('+'):
        return DOCUMENT_LEVEL

    if line.startswith('*'):
        return ARTICLE_LEVEL

    if line.startswith('@'):
        return CLAUSE_LEVEL

    if line.startswith('#'):
        return POINT_LEVEL


def extract_content(line):
    """
    :type line: str
    """
    return line[1:].strip()


def process_file(file):
    file_name = os.path.basename(file)
    lines = open(file, 'r').readlines()

    res = []
    n = len(lines)
    i = 0
    while i < n:
        ref_question = int(re.split(r'\.\s', lines[i])[1].strip())
        i += 1

        queries = []
        while i < n and check_level(lines[i]) == QUERY_LEVEL:
            queries.append(extract_content(lines[i]))
            i += 1

        docs = []
        while i < n and check_level(lines[i]) == DOCUMENT_LEVEL:
            doc = extract_content(lines[i])
            code = re.split(r'\s+', doc)[0]
            i += 1

            articles = []
            while i < n and check_level(lines[i]) == ARTICLE_LEVEL:
                article = extract_content(lines[i])
                i += 1

                clauses = []
                while i < n and check_level(lines[i]) == CLAUSE_LEVEL:
                    clause = extract_content(lines[i])
                    i += 1

                    points = []
                    while i < n and check_level(lines[i]) == POINT_LEVEL:
                        point = extract_content(lines[i])
                        i += 1
                        points.append(point)
                    if len(points) > 0:
                        clauses.append({'title': clause, 'points': points})
                    else:
                        clauses.append({'title': clause})
                if len(clauses) > 0:
                    articles.append({'title': article, 'clauses': clauses})
                else:
                    articles.append({'title': article})
            if len(articles) > 0:
                docs.append({'code': code, 'title': doc, 'articles': articles})
            else:
                docs.append({'code': code, 'title': doc})

        for q in queries:
            obj = {
                'query': q,
                'reference': ref_question,
                'from_file': file_name,
                'documents': docs
            }

            # res.append(obj)
            try:
                collection.insert_one(obj)
            except Exception as e:
                print(e)
                break

        while i < n and lines[i].strip() == '':
            i += 1

        # if i == 100:
        #     break

    # f = open("out.json", 'w')
    # json.dump(res, f, indent=4, ensure_ascii=False)
    # f.close()


if __name__ == '__main__':
    data_dir = '/media/ken/Temp/0MasterThesis/data/validation_data/'
    for path in glob(os.path.join(data_dir, '*.txt')):
        print(path)
        process_file(path)
