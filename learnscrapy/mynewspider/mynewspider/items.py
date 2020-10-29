# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class MynewspiderItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    # 文章标题
    title = scrapy.Field()

    # 时间
    date = scrapy.Field()

    # 来源
    article_source = scrapy.Field()

    # 来源url
    article_source_url = scrapy.Field()

    # 简介
    abstract = scrapy.Field()

    # 正文
    content = scrapy.Field()

    # 编辑
    editor = scrapy.Field()

    # ID
    id = scrapy.Field()

    # 新闻url
    url = scrapy.Field()

    # 文章热度（跟帖+参与数)
    heat = scrapy.Field()

    # ToDo
    # 评论字典
    # comments = scrapy.Field()

