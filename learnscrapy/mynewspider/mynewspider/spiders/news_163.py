import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule

import re, requests, json
from scrapy.selector import Selector 
from mynewspider.items import MynewspiderItem

count = 0
class News163Spider(CrawlSpider):

    # 网易新闻爬虫
    name = 'news_163'

    allowed_domains = ['news.163.com']

    start_urls = ['https://news.163.com/']
    rules = (
        Rule(LinkExtractor(allow=r'https://news\.163\.com/.*$', deny=r'https://.*.163.com/photo.*$'), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        news = Selector(response)
        news_url = response.url
  
        global count
        if news.xpath('//*[@id="epContentLeft"]'):
            
            news_item = MynewspiderItem()

            # 新闻url url
            news_item['url'] = news_url
            
            # 文章标题 title
            titleXpath = '//*[@id="epContentLeft"]/h1/text()'
            get_title(news, titleXpath, news_item)

            # 时间 date
            dateXpath = '//*[@id="epContentLeft"]/div[1]/text()'
            get_date(news, dateXpath, news_item)

            # 来源 article_source
            article_sourceXpath = '//*[@id="epContentLeft"]/div[1]/a[1]/text()'
            get_article_souce(news, article_sourceXpath, news_item)

            # 来源url article_souce_url
            article_source_urlXpath = '//*[@id="epContentLeft"]/div[1]/a[1]/@href'
            get_article_souce_url(news, article_source_urlXpath, news_item)

            # 简介 abstract
            # 正文 content
            # ID id
            contentXpath = '//*[@id="endText"]'
            get_content(news, contentXpath, news_item)
            count += 1
            news_item['id'] = count

            # 编辑 editor
            editorXpath = '//*[@class="ep-editor"]'
            get_editor(news, editorXpath, news_item)

            # 文章热度（跟帖+参与数） heat
            tiecount = int(news.xpath('//*[@class="js-tiecount js-tielink"]/text()').get())
            tiejoin = int(news.xpath('//*[@class="js-tiejoincount js-tielink"]/text()').get())
            news_item['heat'] = tiecount + tiejoin
            yield news_item
            # ToDo
            # 评论字典 comments

        #item['domain_id'] = response.xpath('//input[@id="sid"]/@value').get()
        #item['name'] = response.xpath('//div[@id="name"]').get()
        #item['description'] = response.xpath('//div[@id="description"]').get()


''' 文章标题 title '''
def get_title(news, titleXpath, news_item):
    pass


''' 时间 date '''
def get_date(news, dateXpath, news_item):
    pass


''' 来源 article_source '''
def get_article_souce(news, article_sourceXpath, news_item):
    pass


''' 来源url article_souce_url '''
def get_article_souce_url(news, article_source_urlXpath, news_item):
    pass


'''
    简介 abstract
    正文 content
'''
def get_content(news, contentXpath, news_item):
    pass


''' 编辑 editor '''
def get_editor(news, editorXpath, news_item):
    pass