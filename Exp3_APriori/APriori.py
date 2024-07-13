import itertools
import math
import time

class APriori:
    def __init__(self, baskets: list, support_rate=0.15, confidence_rate=0.30):
        self.baskets = baskets  # 篮子列表
        self.support_rate = support_rate    # 归一化的支持度阈值
        self.support_threshold = support_rate * len(baskets)
        self.confidence_rate = confidence_rate  # 置信度阈值
        self.frequent_itemsets_to_support = dict()
        self.frequent_one_itemsets = list()

    # 获得当前项集中的所有物品列表
    def get_unique_items(self, itemsets: list) -> list:
        items = set()
        for itemset in itemsets:
            items = items.union(itemset)
        return list(items)


    # 获得比当前项集高一阶的候选项集列表
    def get_candidate_itemsets(self, frequent_itemsets: list) -> list:
        candidate_itemsets = list()
        for itemset in frequent_itemsets:
            for item in self.frequent_one_itemsets:
                candidate_itemset = itemset.union(item)
                if len(candidate_itemset) > len(itemset) and candidate_itemset not in candidate_itemsets:
                    candidate_itemsets.append(candidate_itemset)
        return candidate_itemsets

    # 获得当前候选项集中的频繁项集字典
    # 键为项集字符串，值为项集的支持度
    def get_frequent_itemsets(self, candidate_itemsets: list) -> list:
        frequent_itemsets = list()
        for itemset in candidate_itemsets:
            support = 0
            for basket in self.baskets:
                # 计算当前项集在所有篮子中的支持度
                if itemset.issubset(basket):
                    support += 1
            if support >= self.support_threshold:
                frequent_itemsets.append((itemset, support / len(self.baskets)))
                # 字典的键为无序集合转换而来的字符串，需要对先对其进行排序再存储及查找，避免出现集合相同而键不同的问题
                self.frequent_itemsets_to_support[str(set(sorted(itemset)))] = support
        return frequent_itemsets

    # 获得当前频繁项集中的关联规则列表
    def get_association_rules(self, frequent_itemsets: list) -> list:
        association_rules = list()
        for itemset in frequent_itemsets:
            itemset = set(sorted(itemset))
            for i in range(1, math.ceil(len(itemset) / 2) + 1):
                lefts = [set(sorted(itemset)) for itemset in list(itertools.combinations(list(itemset), i))]
                for left in lefts:
                    right = itemset.difference(left)
                    confidence = self.frequent_itemsets_to_support[str(itemset)] / self.frequent_itemsets_to_support[str(left)] * 1.0
                    if confidence >= self.confidence_rate:
                        association_rules.append((left, right, confidence))
        return association_rules



    def run(self):
        start_time = time.time()
        k = 1
        # 初始化得到以单个单词为项集的候选项集列表
        items = self.get_unique_items(self.baskets)
        candidate_itemsets = [{item} for item in items]

        print('APriori starts!')
        while k <= 4:
            frequent_itemsets = self.get_frequent_itemsets(candidate_itemsets)
            if len(frequent_itemsets) == 0:
                # 当前阶数下没有频繁项集，迭代结束
                break
            # 按支持度对项集进行排序
            frequent_itemsets = sorted(frequent_itemsets, key=lambda itemset: itemset[1], reverse=True)

            print(f'Write {k}-itemset starts!')
            with open(f'outputs/{k}-itemset.txt', 'w+', encoding='utf-8') as f:
                f.write(f'total: {len(frequent_itemsets)}\n')
                for itemset in frequent_itemsets:
                    f.write(f'{itemset[0]}: {itemset[1]}\n')

            frequent_itemsets = [itemset[0] for itemset in frequent_itemsets]
            if k == 1:
                self.frequent_one_itemsets = frequent_itemsets
            else:
                association_rules = self.get_association_rules(frequent_itemsets)
                with open(f'outputs/{k}-rules.txt', 'w+', encoding='utf-8') as f:
                    f.write(f'total: {len(association_rules)}\n')
                    for rule in association_rules:
                        f.write(f'{rule[0]} -> {rule[1]}: {rule[2]:.2f}\n')
            print(f'Write {k}-itemset done!')
            candidate_itemsets = self.get_candidate_itemsets(frequent_itemsets)
            k += 1
        print(f'APriori is done with the maximum of k is {k - 1}!')
        end_time = time.time()
        print(f'Run time: {end_time - start_time:.2f}s')

