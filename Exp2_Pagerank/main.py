import numpy as np

# 从现有文件中获取前1000项单词的跳转关系
def load_sorted_words(file='../Exp1_MapReduce/outputs/title_to_keys_raw.txt'):
    with open(file, 'r', encoding='utf-8') as f:
        sorted_words = eval(f.read())
    return sorted_words

# 执行Pagerank过程
def run_page_rank(sorted_words, beta=0.85, epsilon=1e-8):
    M = np.zeros((1000, 1000))
    sorted_words_dict = {sorted_words[i][0]: i for i in range(len(sorted_words))}

    # 根据跳转关系对矩阵M进行初始化及归一化
    for i in range(len(sorted_words)):
        for relation_word in sorted_words[i][1]:
            j = sorted_words_dict.get(relation_word)
            M[j][i] = 1.0 / len(sorted_words[i][1])

    print('Pagerank iteration starts!')
    page_rank = np.full(1000, 0.001)
    random_walk = np.full(1000, 0.001)
    page_rank_delta = float('inf')
    # 幂迭代法计算各单词的Pagerank
    while page_rank_delta > epsilon:
        new_page_rank = np.dot(beta * M, page_rank) + (1 - beta) * random_walk
        page_rank_delta = np.linalg.norm(new_page_rank - page_rank, ord=None)
        page_rank = new_page_rank

    words_page_rank = {sorted_words[i][0]: page_rank[i] for i in range(len(sorted_words))}
    sorted_words_page_rank = sorted(words_page_rank.items(), key=lambda word: word[1], reverse=True)
    if beta < 1:
        with open('outputs/pagerank_with_beta.txt', 'w+', encoding='utf-8') as f:
            for i in range(len(page_rank)):
                f.write(f'{sorted_words_page_rank[i][0]} : {sorted_words_page_rank[i][1]}\n')
    else:
        with open('outputs/pagerank.txt', 'w+', encoding='utf-8') as f:
            for i in range(len(page_rank)):
                f.write(f'{sorted_words_page_rank[i][0]} : {sorted_words_page_rank[i][1]}\n')
    print('Pagerank iteration done!')

    # 检验Pagerank计算是否正确
    print(f'sum of Pagerank: {np.sum(page_rank)}')

if __name__ == '__main__':
    sorted_words = load_sorted_words()
    run_page_rank(sorted_words, beta=1)








