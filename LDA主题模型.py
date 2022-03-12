from gensim import corpora
from gensim.models import LdaModel
from pprint import pprint
import jieba
import re
import logging
import pandas as pd
import numpy as np


if __name__ == "__main__":
    for sheet_num in range(0,3):
        raw_corpus = []  # 原始文章集合
        """
        raw_corpus = [doc1,doc2,...] doc = 将整个文章以字符串的形式输入
        """
        # 读取文章
        df = pd.DataFrame(pd.read_excel("./Corpus/data1.xlsx", engine='openpyxl', sheet_name=sheet_num))
        raw_corpus=df.values
        stop_words = []  # 停用词表
        # 加载百度的停用词表
        with open("stopwords/baidu_stopwords.txt", 'r', encoding='utf-8') as f:
            for line in f:
                stop_words.append(line.strip())
        # 加载自定义的停用词表
        with open("stopwords/myStopWords.txt", 'r', encoding='utf-8') as f:
            for line in f:
                stop_words.append(line.strip())

        corpus = []  # 语料
        """
        corpus = [sentence1,sentence2,...] , sentence = [word1,word2,...]
        """
        for article in raw_corpus:
            article = ''.join(re.findall(r'[\u4e00-\u9fa5]+', article[0]))  # 仅保留中文，如需要数字和英文请注释本行
            corpus.append([item for item in jieba.cut(article) if item not in stop_words])  # 去掉停止词
            # corpus.append([item for item in jieba.cut(article)])

        print(corpus)

        dictionary = corpora.Dictionary(corpus)
        '''
        dictionary.filter_tokens(['都'])  # 该方法可以从字典中删去特定的词,注意若停用词表中已经删去，则这里不需要添加
        dictionary.filter_tokens(['一个'])
        dictionary.filter_tokens(['年'])
        dictionary.filter_tokens(['人'])
        dictionary.compactify()  # 填补因删去词而产生的空白
        '''

        # dictionary.save('lda.dict')  # 存储分完词并删去了停用词的字典


        corpus = [dictionary.doc2bow(sentence) for sentence in corpus]
        '''
        corpora.MmCorpus.serialize('corpus_bow.mm', corpus)  # 存储语料库
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)  # 用于显示训练信息
        '''

        # Set training parameters.
        num_topics = 10  # num_topics – The number of requested latent topics to be extracted from the training corpus.
        chunksize = 2000
        passes = 20
        iterations = 400
        eval_every = None  # Don't evaluate model perplexity, takes too much time.

        # Make a index to word dictionary.
        temp = dictionary[0]  # This is only to "load" the dictionary.必须要有这一步加载，否则报错
        id2word = dictionary.id2token

        # 生成并训练模型
        model = LdaModel(
            corpus=corpus,
            id2word=id2word,
            chunksize=chunksize,
            alpha='auto',
            eta='auto',
            iterations=iterations,
            num_topics=num_topics,
            passes=passes,
            eval_every=eval_every
        )

        # model.save('lda.model')  # 将模型保存到硬盘

        # 打印每篇文章最热门的十个主题
        # top_topics = model.top_topics(corpus=corpus, topn=10)  # topn取想要出现的主题数
        # pprint(top_topics)

        # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
        # avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
        # print('Average topic coherence: %.4f.' % avg_topic_coherence)

        # 参数说明：num_topics代表想从语料库中提取多少个主题，num_words代表每个主题下的关键字数量
        # 注意这里的num_topics不能大于训练模型时设定的num_topics
        for topic in model.print_topics(num_topics=5, num_words=3):
            print(topic[1])
        print()

