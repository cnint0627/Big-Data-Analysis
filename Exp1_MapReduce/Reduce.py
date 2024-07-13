from Map import Map
from threading import Thread

# Reduce节点
class Reduce(Thread):
    # Reduce节点初始化
    def __init__(self, id, inputs: list()):
        super().__init__()
        self.id = id    # 节点编号
        self.inputs = inputs    # Reduce节点上的Map节点输入
        self.output = dict()    # 节点输出

    # Reduce算法执行
    def run(self):
        print(f'Reduce {self.id} starts!')
        words = dict()   # Map节点中每个词汇的出现次数及跳转关系字典

        for pair in self.inputs:
            # title, word = pair[0]
            # if words.get(title) is None:
            #     # 初始化单个词汇的字典
            #     words[title] = {'count': 0, 'relations': list()}
            # if words.get(word) is None:
            #     # 初始化单个词汇的字典
            #     words[word] = {'count': 0, 'relations': list()}
            # # 统计出现次数及跳转关系
            # words[word]['count'] += 1
            # if word not in words[title]['relations']:
            #     words[title]['relations'].append(word)

            title, word = pair[0]
            if words.get(title) is None:
                # 初始化单个词汇的字典
                words[title] = {'count': 0, 'relations': list()}
            if words.get(word) is None:
                # 初始化单个词汇的字典
                words[word] = {'count': 0, 'relations': list()}
            # 根据Map节点Combine过程的结果直接得到单词在单个文档中的出现次数
            words[word]['count'] += pair[1]
            if word not in words[title]['relations']:
                words[title]['relations'].append(word)

        # 排序得到出现次数最多的前1000个词汇作为节点输出
        # self.output = sorted(words.items(), key = lambda word: word[1]['count'], reverse = True)[:1000]
        self.output = words
        print(f'Reduce {self.id} is done!')


