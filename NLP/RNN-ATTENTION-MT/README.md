# 我的工作
- 数据预处理部分input_data.py和一些辅助方法utils.py独自完成
- 模型构建nmt.py主要参考https://github.com/carrie0307/attention-nmt/tree/master/Attention. 按照自己的理解重写部分代码, 使其能和数据预处理的结果相对应; 此外添加了详细的注释
- 改进之处在于
    1. 原来使用的是BasicRNNCell, 训练过程中会梯度爆炸, 改成LSTMCell
    2. 添加dropout
    3. 修改模型评估部分代码, 应用https://github.com/mjpost/sacrebleu中的方法
    4. 添加了记录log的代码

# 模型训练
```
python nmt.py --mode train --learning_rate 0.001 --epochs 10 --batch_size 64 --embedding_dim 300 --hidden_dim 300
```

# 模型预测
```
python nmt.py --mode infer --model_path '2020-12-03 09-34-45' --beam_width 2
```

# 模型评估
```
cat infer.en.tok |sacrebleu ref.en.tok
```

# 其他说明
- 使用的是tensorflow1.15.0版本
- log.txt里记录了训练日志, 其中loss测试了多种超参数也没有降到很低, 没找出原因
- sacrebleu测试的评分很低, 很多翻译的都牛头不对马嘴

# 参数设置
参数设置方法如以下函数所示:
```
def parse_args():
    parser = argparse.ArgumentParser()
    # 训练数据路径
    parser.add_argument('--train_src_path', type = str, default = './data/train.zh.tok')
    parser.add_argument('--train_target_path', type = str, default = './data/train.en.tok')
    # 验证数据路径
    parser.add_argument('--valid_src_path', type = str, default = './data/valid.zh.tok')
    parser.add_argument('--valid_target_path', type = str, default = './data/valid.en.tok')
    # 测试数据路径
    parser.add_argument('--test_src_path', type = str, default = './data/test.zh.tok')
    parser.add_argument('--test_target_path', type = str, default = './data/test.en.tok')
    # 测试结果输出路径
    parser.add_argument('--result_path', type = str, default = './result/translate.en.tok')
    # 日志记录路径
    parser.add_argument('--log_path', type = str, default = 'log.txt')
    # 训练轮数
    parser.add_argument('--epochs', type = int, default = 10)
    # batch_size
    parser.add_argument('--batch_size', type = int, default = 64)
    # embedding_dim
    parser.add_argument('--embedding_dim', type = int, default = 300)
    # hidden_dim
    parser.add_argument('--hidden_dim', type = int, default = 300)
    # learning_rate
    parser.add_argument('--learning_rate', type = float, default = 0.001)
    # mode, 设置是训练模型train, 还是根据模型预测infer
    parser.add_argument('--mode', type = str, default = 'train', help = 'train / infer')
    # beam_width
    parser.add_argument('--beam_width', type = int, default = 2)
    # model_path
    parser.add_argument('--model_path', type = str, default = '')
    return parser.parse_args()
```