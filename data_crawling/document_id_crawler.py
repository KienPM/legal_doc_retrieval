""" Create by Ken at 2020 Feb 12 """
import os
import scrapy
from scrapy import signals
from scrapy.crawler import CrawlerProcess
from pymongo import MongoClient

mongo_client = MongoClient('localhost', 27017)
db = mongo_client['legal-documents']
collection = db['original']
white_list = ['bộ luật', 'luật', 'nghị định', 'thông tư']


def check_type(title):
    """
    :type title: str
    :return:
    """
    title = title.lower()
    for term in white_list:
        if title.startswith(term):
            return True
    return False


class DocumentIdSpider(scrapy.Spider):
    name = "document_id"
    current = 1
    total = 0
    link_template = 'http://vbpl.vn/TW/Pages/vanban.aspx?fromyear=01/01/1945&toyear=31/12/2020&Page={}'
    document_id_list = set()
    filtered_out_document_list = set()

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super(DocumentIdSpider, cls).from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_closed, signal=signals.spider_closed)
        return spider

    def start_requests(self):
        yield scrapy.Request(url=self.link_template.format(self.current), callback=self.parse)

    def parse(self, response):
        content = response.css('div#tabVB_lv1 > div.content')[0]

        if self.current == 1:
            last = content.css('div.paging a::attr(href)').extract()[-1]
            last = int(last[last.rfind('=') + 1:])
            self.total = last

        a_elements = content.css('ul.listLaw > li > div.item > p.title > a')
        for a in a_elements:
            text = a.css('::text').get().strip()
            href = a.css('::attr(href)').get()
            document_id = href[href.rfind('=') + 1:]
            if check_type(text):
                self.document_id_list.add(document_id)
            else:
                self.filtered_out_document_list.add(document_id)

        if self.current < self.total:
            self.current += 1
            yield scrapy.Request(url=self.link_template.format(self.current), callback=self.parse)

    def spider_closed(self, spider):
        self.log('Spider Closed')
        os.makedirs("output", exist_ok=True)

        out_1 = open(os.path.join("output", "document_ids.txt"), "w")
        for id_ in self.document_id_list:
            out_1.write(id_ + '\n')
        out_1.close()

        out_2 = open(os.path.join("output", "filtered_out_document_ids.txt"), "w")
        for id_ in self.filtered_out_document_list:
            out_2.write(id_ + '\n')
        out_2.close()


if __name__ == '__main__':
    process = CrawlerProcess()
    process.crawl(DocumentIdSpider)
    process.start()
