from APriori import  APriori
import itertools
import time

class PCY(APriori):
    def __init__(self, baskets: list, support_rate=0.15, confidence_rate=0.30):
        super().__init__(baskets, support_rate, confidence_rate)

    # 哈希函数，得到项集映射到对应桶的编号
    def hash(self, itemset: set):
        # H({x, y}) = ((order of x) * 10 + (order of y))mod 7
        [x, y] = list(itemset)
        ht_key = (len(x) * 2 + len(y)) % 64
        return ht_key

    # 由篮子生成桶的哈希表，并将结果转换为由01组成的位图
    # 其中1表示频繁桶，0表示非频繁桶
    def generate_buckets_bitmap(self) -> list:
        buckets_ht = [0 for i in range(64)]
        for itemset in self.baskets:
            for generated_itemset in list(itertools.combinations(list(itemset), 2)):
                ht_key = self.hash(set(generated_itemset))
                buckets_ht[ht_key] += 1
        # 得到桶的位图
        buckets_bitmap = [1 if buckets_ht[i] >= self.support_threshold else 0 for i in range(64)]
        return buckets_bitmap

    def get_candidate_itemsets(self, frequent_itemsets: list, buckets_bitmap=None) -> list:
        candidate_itemsets_raw = super().get_candidate_itemsets(frequent_itemsets)
        if buckets_bitmap is None:
            return candidate_itemsets_raw
        candidate_itemsets = list()
        for candidate_itemset in candidate_itemsets_raw:
            ht_key = self.hash(candidate_itemset)
            if buckets_bitmap[ht_key] == 1:
                candidate_itemsets.append(candidate_itemset)
        return candidate_itemsets
    def run(self):
        start_time = time.time()
        k = 1
        # 初始化得到以单个单词为项集的候选项集列表
        items = self.get_unique_items(self.baskets)
        candidate_itemsets = [{item} for item in items]

        print('PCY starts!')
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
            buckets_bitmap = None
            if k == 1:
                self.frequent_one_itemsets = frequent_itemsets
                # New in PCY
                buckets_bitmap = self.generate_buckets_bitmap()
                print(f'Buckets bitmap: {buckets_bitmap}')
            else:
                association_rules = self.get_association_rules(frequent_itemsets)
                with open(f'outputs/{k}-rules.txt', 'w+', encoding='utf-8') as f:
                    f.write(f'total: {len(association_rules)}\n')
                    for rule in association_rules:
                        f.write(f'{rule[0]} -> {rule[1]}: {rule[2]:.2f}\n')
            print(f'Write {k}-itemset done!')
            candidate_itemsets = self.get_candidate_itemsets(frequent_itemsets, buckets_bitmap)
            k += 1
        print(f'PCY is done with the maximum of k is {k - 1}!')
        end_time = time.time()
        print(f'Run time: {end_time - start_time:.2f}s')


