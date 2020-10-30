# 学习使用Scrapy爬虫
- 创建项目

    ```scrapy startproject tutorial```
    
    这将创建一个```tutorial```项目，包含
    ```
        tutorial/
        scrapy.cfg            # deploy configuration file

        tutorial/             # project's Python module, you'll import your code from here
            __init__.py

            items.py          # project items definition file

            middlewares.py    # project middlewares file

            pipelines.py      # project pipelines file

            settings.py       # project settings file

            spiders/          # a directory where you'll later put your spiders
                __init__.py
    ```

# 创建一个基于CrawlSpider的爬虫文件
```
scrapy genspider -t crawl SpiderName www.xxx.com
```
# 依赖库
pip install scrapy


# Docker

```
sudo yum remove docker \
                  docker-client \
                  docker-client-latest \
                  docker-common \
                  docker-latest \
                  docker-latest-logrotate \
                  docker-logrotate \
                  docker-engine
```

```
$ sudo yum install -y yum-utils

$ sudo yum-config-manager \
    --add-repo \
    https://download.docker.com/linux/centos/docker-ce.repo

$ sudo yum install docker-ce docker-ce-cli containerd.io

$ sudo systemctl start docker
```

# 参考
>https://www.osgeo.cn/scrapy/intro/tutorial.html

>https://github.com/Wooden-Robot/scrapy-tutorial

>https://docs.docker.com/engine/install/centos/


