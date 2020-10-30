# -*- coding: utf-8 -*-
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from scrapy_splash import SplashRequest
from scrapy_splash import SplashTextResponse, SplashJsonResponse, SplashResponse
from scrapy.http import HtmlResponse


class BaiduSpider(CrawlSpider):
    name = 'baidu'  # 忽略这个任务名称
    # allowed_domains = ['baidu.com']
    start_urls = ['http://www.yuci.gov.cn/index.html']

    rules = (
        Rule(LinkExtractor(
            restrict_xpaths="//div[@class='bd']/div[@class='conWrap']/div/ul/li[last()]/a[contains(text(),'查看更多')]"),
            callback='parse_item',follow=True),
        Rule(LinkExtractor(restrict_xpaths="//ul[@class='wz_list']/li//a"), callback='parses_item'),
    )

    def start_requests(self):
        for url in self.start_urls:
            print("进入首页")
            print(url)
            yield scrapy.Request(url)

    def _build_request(self, rule, link):
        print("进入下")
        # print(link.url)
        # print(link.text)
        r = SplashRequest(url=link.url, callback=self._response_downloaded, args={"wait": 0.5})
        r.meta.update(rule=rule, link_text=link.text)
        return r

    def _requests_to_follow(self, response):

        # if not isinstance(response, HtmlResponse):
        #     return
        seen = set()
        for n, rule in enumerate(self._rules):
            links = [lnk for lnk in rule.link_extractor.extract_links(response)
                     if lnk not in seen]
            if links and rule.process_links:
                links = rule.process_links(links)
            # print("奈斯")
            print(len(links))
            for link in links:
                # print(link)
                seen.add(link)
                # print("啊哈")
                r = self._build_request(n, link)
                yield rule.process_request(r)

    def parse_item(self, response):
        i = {}
        print("进入列表页")
        print(response)
        return i

    def parses_item(self, response):
        # print("进入详情页")
        name = response.xpath("//h1/text()").extract()
        print(name)
        return ""