import os
import re
from threading import Thread

# Map节点
class Map(Thread):
    # Map节点初始化
    def __init__(self, id, root_dir: str, key_words: dict):
        super().__init__()
        self.id = id    # 节点编号
        self.root_dir = root_dir    # 节点数据的根目录
        self.key_words = key_words  # 待统计的词汇
        self.output = list()    # 节点输出

    # 文本分词处理
    def strip_punctuation(self, text: str) -> list:
        content = re.split(r'[\W+]', text)
        return content

    # Map算法执行
    def run(self):
        print(f'Map {self.id} starts!')
        # 获取根目录下的全部文件
        file_names = [f for f in os.listdir(self.root_dir)]
        # print(file_names)
        # 依次读取文件内容
        current = 0
        total = len(file_names)
        pairs = dict()
        for file_name in file_names:
            current += 1
            # if current % 10 == 0:
            #     print(f'{self.id} {current} / {total}')
            title = file_name.replace('.txt', '')
            with open(f'{self.root_dir}/{file_name}', 'r', encoding = 'utf-8') as f:
                content = f.read()
                # 对单个文本进行分词处理
                content = self.strip_punctuation(content)
                # 遍历分词后的单词列表，对满足条件的词汇添加到Map节点的输出中
                for word in content:
                    word = word.lower()
                    if self.key_words.get(word) == 1:
                        # Combine操作，对同一title下的相同word进行合并
                        if pairs.get((title, word)) is None:
                            pairs[(title, word)] = 0
                        pairs[(title, word)] += 1
        self.output = list(pairs.items())
        # 将每个Map的output分成三份，为后续Shuffle工作做准备
        output_size = len(self.output)
        self.output = [self.output[: (output_size//3)], self.output[(output_size//3): (output_size//3*2)], self.output[(output_size//3*2):]]
        print(f'Map {self.id} is done!')

if __name__ == '__main__':
    # 读取key_word列表
    key_words_file = open('../words.txt', 'r')
    key_words = key_words_file.read().splitlines()
    key_words_file.close()

    # 将key_word列表转换为字典，后续可加快比对速度
    key_words = {key_word: 1 for key_word in key_words}
    # print(key_words)
    map1 = Map(id = 2, root_dir = '../source_data/folder_3', key_words = key_words)
    map1.start()
    # print(map1.output)

