# -*- coding: utf-8 -*-
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule

import re, requests, json
from scrapy.selector import Selector 
from mynewspider.items import MynewspiderItem


'''
# 使用selenium渲染js，速度太慢，放弃
from selenium import webdriver
from selenium.webdriver.chrome.options import Options # 使用无头浏览器
# 无头浏览器设置
chrome_options = Options()
chrome_options.add_argument("headless")
chrome_options.add_argument("disable-gpu")
'''


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
            '''
            # 使用selenium渲染js，速度太慢，放弃
            browser = webdriver.Chrome(chrome_options=chrome_options)
            browser.implicitly_wait(10)
            browser.get(news_url)
            heat = browser.find_element_by_xpath('//*[@id="post_comment_area"]/div[2]/div[2]/a').text
            print("*****************************", heat)
            browser.close()
            '''

            
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
            get_article_source(news, article_sourceXpath, news_item)

            # 来源url article_souce_url
            article_source_urlXpath = '//*[@id="epContentLeft"]/div[1]/a[1]/@href'
            get_article_source_url(news, article_source_urlXpath, news_item)

            # 简介 abstract
            # 正文 content
            # ID id
            contentXpath = '//*[@id="endText"]'
            get_content(news, contentXpath, news_item)
            count += 1
            news_item['id'] = count

            # 编辑 editor
            editorXpath = '//*[@class="ep-editor"]/text()'
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
    try:
        news_title = news.xpath(titleXpath).extract()[0]
        news_title = news_title.replace('\n', '')
        news_title = news_title.replace('\r', '')
        news_title = news_title.replace('\t', '')
        news_title = news_title.replace(' ', '')
        news_item['title'] = news_title
    except:
        news_item['title'] = ''


''' 时间 date '''
def get_date(news, dateXpath, news_item):
    try:
        news_date = news.xpath(dateXpath).extract()[0]
        pattern = re.compile("(\d.*\d)")  # 正则匹配新闻时间
        news_datetime = pattern.findall(news_date)[0]
        news_item['date'] = news_datetime
    except:
        news_item['date'] = ''


''' 来源 article_source '''
def get_article_source(news, article_sourceXpath, news_item):
    try:
        article_source = news.xpath(article_sourceXpath).get()
        news_item['article_source'] = article_source
    except:
        news_item['article_source'] = ''


''' 来源url article_source_url '''
def get_article_source_url(news, article_source_urlXpath, news_item):
    try:
        article_source_url = news.xpath(article_source_urlXpath).get()
        news_item['article_source_url'] = article_source_url
    except:
        news_item['article_source_url'] = ''


'''
    简介 abstract
    正文 content
'''
def get_content(news, contentXpath, news_item):
    try:
        content_data = news.xpath(contentXpath )
        article_content = content_data.xpath('string(.)').extract()[0]
        article_content = str_replace(article_content)
        news_item['content'] = article_content
        # 匹配新闻简介，前100个字
        try:
            abstract = article_content[0:100]
            news_item['abstract'] = abstract
        except:
            news_item['abstract'] = article_content
    except:
        news_item['content'] = ''
        news_item['abstract'] = ''

''' 编辑 editor '''
def get_editor(news, editorXpath, news_item):
    try:
        editor_data = news.xpath(editorXpath).get()
        pattern = re.compile('：(.*)')
        editor = pattern.findall(editor_data)[0]
        news_item['editor'] = editor
    except:
        news_item['editor'] = ''


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
    # To Do 评论字典
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

