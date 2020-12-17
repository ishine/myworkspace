# 自然语言处理作业--关系抽取(Relation Extraction)

## 数据集
完全监督的关系抽取数据集SemEval-2010 Task8

### 训练集数据

- ```data/TRAIN_FILE.TXT```
- 8000个训练样例
- 预处理后为```data/train.json```

### 测试集数据
- ```data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT```
- 2717个测试样例
- 预处理后为```data/test.json```

## 评价指标
测评指标使用的是官方给出的scorer

- ```data/SemEval2010_task8_scorer-v1.2/semeval2010_task8_scorer-v1.2.pl```

## 预训练词向量
使用GloVe的预训练词向量

位于embedding/glove.6B.300d.txt

- ```http://nlp.stanford.edu/data/glove.6B.zip/```
