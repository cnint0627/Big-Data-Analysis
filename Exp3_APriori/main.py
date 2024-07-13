from APriori import APriori
from PCY import PCY
import pickle

# 从现有文件中获取前1000项单词的跳转关系
def load_sorted_words(file='../Exp1_MapReduce/outputs/title_to_keys_raw.txt'):
    with open(file, 'r', encoding='utf-8') as f:
        sorted_words = eval(f.read())
    return sorted_words

# 将单词跳转关系列表转化成桶的形式
def get_baskets_from_sorted_words(sorted_words: list):
    baskets = list()
    for word in sorted_words:
        basket = {word[0]}
        basket = basket.union(word[1])
        baskets.append(basket)
    return baskets

def load_datasets(file='top_keywords.pkl'):
    with open(file, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def get_baskets_from_datasets(dataset: tuple):
    baskets = list()
    for word in dataset[1].items():
        basket = {word[0]}
        basket = basket.union(word[1])
        baskets.append(basket)
    return baskets

if __name__ == '__main__':
    dataset = load_datasets()
    baskets = get_baskets_from_datasets(dataset)
    print('Please choose an algorithm to run:')
    print('1. APriori\t\t2. PCY')
    option = input()
    if option == '1':
        apriori = APriori(baskets=baskets)
        apriori.run()
    elif option == '2':
        pcy = PCY(baskets=baskets)
        pcy.run()
    else:
        print('Invalid option!')