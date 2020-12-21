""" Create by Ken at 2020 Feb 12 """
import os
import argparse
from datetime import datetime
import logging
import re
import requests

from bs4 import BeautifulSoup
import scrapy
from scrapy import signals, Request
from scrapy.crawler import CrawlerProcess
from pymongo import MongoClient

from utils.string_utils import compound_unicode

os.makedirs('logs', exist_ok=True)
logging.getLogger('scrapy').propagate = False
logging.getLogger('urllib3.connectionpool').propagate = False
logging.basicConfig(filename="logs/runtime_{}.log".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S")),
                    format='%(asctime)-15s %(name)s - %(levelname)s %(message)s')
logger = logging.getLogger("crawler")
logger.setLevel(logging.WARNING)

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
    default='raw',
    help='MongoDB collection name'
)
arg_parser.add_argument(
    '--doc_list_file',
    type=str,
    default='document_to_crawl.txt',
    help='Path to file containing document list'
)
args = arg_parser.parse_args()

mongo_client = MongoClient(args.db_host, args.db_port)
db = mongo_client[args.db_name]
collection = db[args.db_collection]

clause_re = re.compile(r'^điều\s*([0-9]+|[MDCLXVI]+)\s*[.:)/\-]|^điều\s*([0-9]+|[MDCLXVI]+)$', re.IGNORECASE)


def normalize_doc_code(doc_code):
    """
    Eliminate space before and after '-', '/'
    E.g. '13/2015/TT- BVHTTDL' => '13/2015/TT-BVHTTDL'; '116 /2002/TT-BTC' => '116/2002/TT-BTC'
    :type doc_code: str
    :rtype: str
    """
    doc_code = re.sub(r'\s*([-/])\s*', r'\g<1>', doc_code)
    return doc_code


def normalize_article_name(article_name):
    """
    Eliminate punctuation
    :type article_name: str
    :rtype: str
    """
    article_name = re.sub(r'[.:)/\-]', '', article_name).strip()
    article_name = article_name.replace(u'\xa0', u' ')
    return article_name


def crawl_properties(id_):
    properties = {}
    r = requests.get('http://vbpl.vn/tw/Pages/vbpq-thuoctinh.aspx?dvid=13&ItemID=' + id_)
    if r.status_code >= 400:
        raise Exception('Connection exception, status code: {}'.format(r.status_code))

    html = BeautifulSoup(r.content, 'html.parser')
    table_data = html.select('div.vbProperties > table > tbody > tr > td')

    properties["ten_van_ban"] = table_data[0].text.strip()
    properties["so_hieu"] = normalize_doc_code(table_data[2].text.strip())
    properties["ngay_ban_hanh"] = table_data[4].text.strip()
    properties["loai_van_ban"] = table_data[6].text.strip()
    properties["ngay_hieu_luc"] = table_data[8].text.strip()

    for i in range(9, len(table_data)):
        if table_data[i].text.strip() == 'Ngành':
            properties["nganh"] = table_data[i + 1].text.strip()

        if table_data[i].text.strip() == 'Lĩnh vực':
            properties["linh_vuc"] = table_data[i + 1].text.strip()

        if table_data[i].text.strip() == 'Cơ quan ban hành/ Chức danh / Người ký':
            properties["noi_ban_hanh"] = table_data[i + 1].text.strip()
            properties["nguoi_ky"] = table_data[i + 3].text.strip()

        if table_data[i].text.strip() == 'Phạm vi':
            properties["pham_vi"] = table_data[i + 1].text.strip()

        if table_data[i].text.strip().startswith('Tình trạng hiệu lực: '):
            properties["hieu_luc"] = table_data[i].text.strip().replace('Tình trạng hiệu lực: ', '').strip()

    # --------------------------------------------------------------------------------------------------

    files = html.select('#divShowDialogDownload .fileAttack li > span > a')
    properties["file"] = []
    for a in files:
        link = a['href'].split("'")[-2]
        if link.split('.')[-1] == 'pdf':
            link = 'https://bientap.vbpl.vn/' + link
        else:
            link = 'http://vbpl.vn/tw/_layouts/15/WopiFrame.aspx?sourcedoc=' + link

        properties["file"].append(link)

    return properties


def get_all_id(html):
    all_id = [a['href'].split('=')[-1] for a in html.select('.content ul > li > a') if a['href'] != '#']
    return all_id


def crawl_related_documents(id_):
    r = requests.get('http://vbpl.vn/TW/Pages/vbpq-luocdo.aspx?ItemID=' + id_)
    if r.status_code >= 400:
        raise Exception('Connection exception, status code: {}'.format(r.status_code))

    related_documents = {"van_ban_duoc_huong_dan": [],
                         "van_ban_hien_thoi": [],
                         "van_ban_huong_dan": [],
                         "van_ban_het_hieu_luc": [],
                         "van_ban_can_cu": [],
                         "van_ban_quy_dinh_het_hieu_luc": [],
                         "van_ban_bi_het_hieu_luc_mot_phan": [],
                         "van_ban_dan_chieu": [],
                         "van_ban_quy_dinh_het_hieu_luc_mot_phan": [],
                         "van_ban_bi_dinh_chi": [],
                         "van_ban_lien_quan_khac": [],
                         "van_ban_dinh_chi": [],
                         "van_ban_bi_dinh_chi_mot_phan": [],
                         "van_ban_dinh_chi_mot_phan": [],
                         "van_ban_duoc_bo_sung": [],
                         "van_ban_bo_sung": [],
                         "van_ban_duoc_sua_doi": [],
                         "van_ban_sua_doi": []}

    try:
        html = BeautifulSoup(r.content, 'html.parser')
        table_data = html.select('.vbLuocDo table > tr > td')
        related_documents["van_ban_duoc_huong_dan"] = get_all_id(table_data[5])
        related_documents["van_ban_hien_thoi"] = get_all_id(table_data[7])
        related_documents["van_ban_huong_dan"] = get_all_id(table_data[9])
        related_documents["van_ban_het_hieu_luc"] = get_all_id(table_data[10])
        related_documents["van_ban_can_cu"] = get_all_id(table_data[12])
        related_documents["van_ban_quy_dinh_het_hieu_luc"] = get_all_id(table_data[14])
        related_documents["van_ban_bi_het_hieu_luc_mot_phan"] = get_all_id(table_data[15])
        related_documents["van_ban_dan_chieu"] = get_all_id(table_data[17])
        related_documents["van_ban_quy_dinh_het_hieu_luc_mot_phan"] = get_all_id(table_data[19])
        related_documents["van_ban_bi_dinh_chi"] = get_all_id(table_data[20])
        related_documents["van_ban_lien_quan_khac"] = get_all_id(table_data[22])
        related_documents["van_ban_dinh_chi"] = get_all_id(table_data[24])
        related_documents["van_ban_bi_dinh_chi_mot_phan"] = get_all_id(table_data[5])
        related_documents["van_ban_dinh_chi_mot_phan"] = get_all_id(table_data[29])
        related_documents["van_ban_duoc_bo_sung"] = get_all_id(table_data[30])
        related_documents["van_ban_bo_sung"] = get_all_id(table_data[34])
        related_documents["van_ban_duoc_sua_doi"] = get_all_id(table_data[35])
        related_documents["van_ban_sua_doi"] = get_all_id(table_data[39])
    except IndexError:
        pass

    return related_documents


def extract_text(paragraphs):
    """
    Extracts text from bs4.element.Tag
    Ignore empty lines and text in tag align center but isn't a clause
    Eliminates appendix part
    Get first clause position
    :return: Paragraphs, first clause position (-1 id not exists)
    :rtype: list[str], int
    """
    n = len(paragraphs)
    result = []
    first_clause_pos = -1
    ignored_lines = 0
    for i in range(n):
        paragraph = paragraphs[i]
        text = paragraph.text.strip()
        lower_text = text.lower()

        if first_clause_pos == -1 and clause_re.search(lower_text) is not None:
            first_clause_pos = i - ignored_lines

        if lower_text.startswith('phụ lục'):
            # Check if 'phụ lục...' contained in <strong> tag or <b> tag
            strong_tags = paragraph.findChildren("strong")
            b_tags = paragraph.findChildren("b")
            if len(strong_tags) > 0 or len(b_tags) > 0:
                break

        if clause_re.search(lower_text) is not None or (
                text != '' and not (paragraph.get('align') and paragraph.get('align').lower() == 'center')):
            result.append(compound_unicode(text))
        else:
            ignored_lines += 1

    return result, first_clause_pos


def parse_clauses(paragraphs, first_clause_pos):
    result = []

    n = len(paragraphs)
    i = first_clause_pos
    while i < n:
        clause = {}
        found = clause_re.search(paragraphs[i])
        clause["ten_dieu"] = normalize_article_name(found.group())
        clause["tieu_de"] = paragraphs[i][found.end():].strip()
        clause_content = []
        i += 1

        while i < n and clause_re.search(paragraphs[i]) is None:
            clause_content.append(paragraphs[i])
            i += 1
        clause["noi_dung"] = '\n'.join(clause_content)
        result.append(clause)

    return result


def get_year(date_string):
    """
    :type date_string: str
    :rtype: int
    """
    return int(re.split(r'[/\-]', date_string)[-1].strip())


def delete_document_title(paragraphs, doc_name):
    """
    :type paragraphs: list[str]
    :type doc_name: str
    :rtype: list[str]
    """
    n = len(paragraphs)
    i = 0

    doc_name = doc_name.lower()
    while i < n and paragraphs[i].lower() != doc_name:
        i += 1

    if i < n:
        paragraphs = paragraphs[i + 1:]
    return paragraphs


class DocumentSpider(scrapy.Spider):
    name = "document"
    crawled_docs = {}
    base_url = 'http://vbpl.vn/tw/Pages/vbpq-toanvan.aspx?ItemID='

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super(DocumentSpider, cls).from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_closed, signal=signals.spider_closed)
        return spider

    def start_requests(self):
        id_list = [line.strip() for line in open(args.doc_list_file, 'r').readlines() if line != '\n']
        # id_list = ['71274']

        # crawl data for each link
        total = len(id_list)
        print('Total: {}'.format(total))
        for index, id_ in enumerate(id_list):
            if (index + 1) % 100 == 0:
                print('{} / {}'.format(index + 1, total))
            link = self.base_url + id_
            yield Request(url=link, callback=self.parse, meta={"id_": id_})
            # break

    def parse(self, response):
        id_ = response.meta['id_']
        try:
            if not response.url.startswith(self.base_url):  # In case the document is not available in HTML
                logger.warning('Ignore document with id: {}, reason: document is not available in HTML'.format(id_))
                return

            document = {"id": id_}

            properties = crawl_properties(id_)
            if properties["hieu_luc"].lower() in ['hết hiệu lực toàn bộ', 'ngưng hiệu lực']:
                logger.warning('Ignore document with id: {}, reason: document is expired'.format(id_))
                return
            for key in properties:
                document[key] = properties[key]

            related_documents = crawl_related_documents(id_)
            document["van_ban_lien_quan"] = related_documents

            soup = BeautifulSoup(response.body, "lxml")
            paragraphs = soup.select("#toanvancontent > p, h1, h2, h3")
            paragraphs, first_clause_pos = extract_text(paragraphs)
            if len(paragraphs) < 3:
                paragraphs = soup.select("#toanvancontent > div > p, h1, h2, h3")
                paragraphs, first_clause_pos = extract_text(paragraphs)
            if len(paragraphs) < 3:
                paragraphs = soup.select("#toanvancontent > div, p, h1, h2, h3")
                paragraphs, first_clause_pos = extract_text(paragraphs)
            if len(paragraphs) < 3:
                logger.warning('Ignore document with id: {}, reason: document is empty'.format(id_))
                return

            if first_clause_pos != -1:
                clauses = parse_clauses(paragraphs, first_clause_pos)
                document["cac_dieu"] = clauses
                collection.insert_one(document)
            else:
                # If document is not clearly divided into 'Điều 1, Điều 2,...'
                # - If document has been published in 2000 or earlier then ignore
                # - Else merge all content to "clause 0" (Here named '#0')
                year = get_year(document["ngay_ban_hanh"])
                if year > 2000:
                    paragraphs = delete_document_title(paragraphs, document["ten_van_ban"])
                    document["cac_dieu"] = [
                        {
                            "ten_dieu": "#0",
                            "tieu_de": "#0",
                            "noi_dung": ' '.join(paragraphs)
                        }
                    ]
                    collection.insert_one(document)
                else:
                    logger.warning('Ignore document with id: {}, reason: format and published time'.format(id_))
        except Exception as e:
            logger.error('Exception when process document with id: {}'.format(id_), exc_info=True)

    def spider_closed(self, spider):
        self.log('Spider Closed')


if __name__ == '__main__':
    process = CrawlerProcess()
    process.crawl(DocumentSpider)
    process.start()
