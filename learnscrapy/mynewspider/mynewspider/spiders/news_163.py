# -*- coding: utf-8 -*-
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule

import re, requests, json
from scrapy.selector import Selector 
from mynewspider.items import MynewspiderItem
from scrapy_splash import SplashRequest
from scrapy_splash import SplashTextResponse, SplashJsonResponse, SplashResponse
from scrapy.http import HtmlResponse


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

            # 文章热度（跟帖） heat
            # 评论字典 comments
            commentinfoXpath = '//*[@id="post_comment_area"]/script[5]/text()'
            comment_url = get_comment_url(news, commentinfoXpath)
            get_comment(comment_url, news_item)
            yield news_item

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


''' 评论url '''
def get_comment_url(news, commentinfoXpath):
    # 提取新闻评论路径
    news_info = news.xpath(commentinfoXpath)
    news_info_text = news_info.extract()[0]
    # 正则寻找
    pattern_productKey = re.compile("\"productKey\" :.*")
    productKey_text = pattern_productKey.findall(news_info_text)[0]
    productKey = re.findall(r"\"productKey\".*\"(.*)\"", productKey_text)
    pattern_docId = re.compile("\"docId\" :.*")
    docId_text = pattern_docId.findall(news_info_text)[0]
    docId = re.findall(r"\"docId\".*\"(.*)\"", docId_text)
    comment_url = 'http://comment.news.163.com/api/v1/products/' + productKey[0] + '/threads/' + docId[0] + '/comments/newList?offset=0'
    return comment_url


''' 
    评论处理函数 
    # 文章热度（跟帖） heat
    # 评论字典 comments
    comments_dict = {
        'id': ,
        'username': ,
        'date_time': ,
        'content': 
    }
'''
def get_comment(comment_url, news_item):
    comment_data = requests.get(comment_url).text
    js_comment = json.loads(comment_data)
    # 文章热度（跟帖） heat
    heat = js_comment['newListSize']
    news_item['heat'] = heat
    '''
    comments = []
    comment_id = 0
    try:
        comment_data = requests.get(comment_url).text
        js_comment = json.loads(comment_data)
        try:
            # 文章热度（跟帖） heat
            heat = js_comment['newListSize']
            news_item['heat'] = heat
            
            js_comments = js_comment['comments']
            for each, value in js_comment['comments'].items():
                comment_id += 1
                comments_dict = {}
                print(value)
                
                # 评论id
                comments_dict['id'] = comment_id
                # 评论用户名
                try:
                    comments_dict['username'] = value['user']['nickname']
                except:
                    comments_dict['username'] = '匿名用户'
                # 评论时间， datetime格式
                try:
                    date_time = value['createTime']
                    comments_dict['date_time'] = date_time
                except:
                    comments['date_time'] = news_item['date']
                # 评论内容
                ori_content = value['content']
                content = str_replace(ori_content)
                comments_dict['content'] = content
                comments.append(comments_dict)
            if comments:
                print(len(comments), "*************************", heat)
                return heat, comments
            else:
                return 0, ''
            
        except:
            return 0, ''
    except:
        return 0, ''
    '''


''' 字符过滤函数 '''
def str_replace(content):
    try:
        article_content = re.sub('[\sa-zA-Z\[\]!/*(^)$%~@#…&￥—+=_<>.{}\'\-:;"‘’|]', '', content)
        return article_content
    except:
        return content

