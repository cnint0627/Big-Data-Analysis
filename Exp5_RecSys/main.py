import numpy as np
import pandas as pd
import time
import warnings
from CollaborativeFiltering import CollaborativeFiltering
from ContentBased import ContentBased

def load_dataset(file: str) -> pd.DataFrame:
    dataset = pd.read_csv(file)
    return dataset

if __name__ == '__main__':
    # 忽略所有警告
    warnings.filterwarnings('ignore')

    # 导入数据集
    train_set = load_dataset('data/train_set.csv').to_numpy()
    test_set = load_dataset('data/test_set.csv').to_numpy()
    anime_set = load_dataset('data/anime.csv')

    # 交互界面
    print("请选择推荐系统模型：")
    print("1. 基于用户-用户协同过滤的推荐")
    print("2. 基于内容的推荐")
    model_option = int(input())
    print("请选择模型算法：")
    print("1. 普通版")
    print("2. MinHash版")
    minhash_option = int(input())
    minhash_option = True if minhash_option == 2 else False

    start_time = time.time()
    if model_option == 1:
        cf = CollaborativeFiltering(train_set, minhash=not minhash_option)
        cf.fit()
        cf.recommend(629, 150)
        sse = 0
        for data in test_set:
            user_id, anime_id, rating = data
            score = cf.predict(user_id, anime_id, 150)[0]
            score = 5 if np.isnan(score) else score
            sse += (rating - score) ** 2
        print(f'CF  {"MinHash" if minhash_option else "基础"}版SSE: {sse}')
    if model_option == 2:
        cb = ContentBased(train_set, anime_set, minhash=not minhash_option)
        cb.fit()
        cb.recommend(629)
        sse = 0
        for data in test_set:
            user_id, anime_id, rating = data
            score = cb.predict(user_id, anime_id)
            score = 5 if np.isnan(score) else score
            sse += (rating - score) ** 2
        print(f'Content-Based  {"MinHash" if minhash_option else "基础"}版SSE: {sse}')

    end_time = time.time()
    print(f'总时间：{end_time - start_time:.2f}s')

