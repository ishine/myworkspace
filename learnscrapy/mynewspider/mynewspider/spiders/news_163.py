import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule

import re, requests, json
from scrapy.selector import Selector 

count = 0
all_count = 0
class News163Spider(CrawlSpider):

    # 网易新闻爬虫
    name = 'news_163'

    allowed_domains = ['news.163.com']

    start_urls = ['https://news.163.com/']
    rules = (
        Rule(LinkExtractor(allow=r'https://news\.163\.com/.*$', deny=r'https://.*.163.com/photo.*$'), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        article = Selector(response)
        article_url = response.url
        articleXpath = '//*[@id="epContentLeft"]'
        global count
        global all_count
        all_count += 1
        if article.xpath(articleXpath):
            count += 1
            print("***********************************************************************",article_url, count, all_count)
        #item['domain_id'] = response.xpath('//input[@id="sid"]/@value').get()
        #item['name'] = response.xpath('//div[@id="name"]').get()
        #item['description'] = response.xpath('//div[@id="description"]').get()
