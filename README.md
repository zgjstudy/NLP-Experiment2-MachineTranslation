## 概览

本次实验我使用transformer模型进行机器翻译，其中encoder部分使用了bert预训练模型。

模型训练batch size为600，训练了200 epoch。训练的时候没有做验证和early stopping，所以不太清楚训练到什么程度了（看结果并没有训练到位）

模型预测使用beam search，beam size为4，length penalty $\alpha=0.6$ 。由于我写的代码可能效率不太高，预测一条数据需要35秒左右，验证集的7万条数据实在是算不完了，所以我就先预测了4308条。最后得到的BLEU为0.197，计算过程见 [bleu.ipynb](./bleu.ipynb)。

## 模型训练

先下载Google的bert预训练模型：[下载地址](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)

然后配置 [config.py](config.py) 文件，最后运行 [train.py](train.py) 即可开始训练。

## 代码结构

[tokenizer.py](./tokenizer.py) 使用FullTokenizer和tfds.features.text.SubwordTextEncoder作为中英文的tokenizer，将文本数据编码为token ids。

[prepare_dataset.py](./prepare_dataset.py) 对数据根据中英文的长度比做一下简单的筛选，然后将token ids构建为tf.data.Dataset。

[config.py](config.py) 设置了模型训练时的各种参数。

[train.py](train.py) 使用tf.distribute.MirroredStrategy进行分布式训练。





