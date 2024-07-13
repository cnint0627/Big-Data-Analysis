import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from RecommendSystem import RecommendSystem
from MinHash import *

class ContentBased(RecommendSystem):
    def __init__(self, train, animes, minhash=False):
        super().__init__(train, minhash)
        self.animes = animes
        self.id_to_idx = dict()
        self.tfidf_matrix = None
        self.similarity_matrix = None

    # 计算动漫相似度矩阵
    def build_similarity_matrix(self):
        if self.minhash:
            signature_matrix = ContentBasedMinHash.build_signature_matrix(self.tfidf_matrix)
            self.similarity_matrix = ContentBasedMinHash.build_similarity_matrix(signature_matrix)
        else:
            #利用余弦相似度计算动漫之间的相似度矩阵
            self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    # 生成动漫tf-idf矩阵
    def build_tfidf_matrix(self):
        tfidf = TfidfVectorizer()
        self.tfidf_matrix = tfidf.fit_transform(self.animes['Genres'].tolist()).toarray()

    # 构造基于内容的推荐系统模型
    def fit(self):
        print('模型构造开始')
        # 构建id到index的字典
        for i in range(len(self.animes)):
            self.id_to_idx[self.animes['Anime_id'][i]] = i
        self.build_tfidf_matrix()
        self.build_utility_matrix()
        self.build_similarity_matrix()
        print('模型构造完毕')


    # 对指定用户的指定动漫进行预测
    def predict(self, user_id: int, anime_id: int) -> float:
        # 选取用户打过分的动漫
        rated_id = np.where(~np.isnan(self.utility_matrix[:, user_id]))[0]
        # 打过分的所有动漫的分值
        rated_score = self.utility_matrix[rated_id, user_id]
        distances = self.similarity_matrix[self.id_to_idx[anime_id]]  # anime_id 与其它动漫的相似度
        # 计算集合
        similarity_dict = {}
        for i in range(len(rated_id)):
            cosine = distances[self.id_to_idx[rated_id[i]]]
            similarity_dict[i] = cosine
        if len(similarity_dict.keys()):  # 计算集合不为空，则计算加权预测值
            score_sum, sim_sum = 0, 0
            for k, v in similarity_dict.items():
                score_sum += rated_score[k] * v
                sim_sum += v
            return score_sum / sim_sum
        else:  # 计算集合为空，则用平均值当做预测
            return np.mean(rated_score)

    # 对指定用户未评分的动漫进行推荐
    def recommend(self, user_id):
        print('用户推荐开始')
        predictions = dict()
        # 对用户每个未评分项目进行预测
        for anime_id in range(1, self.anime_nums + 1):
            if np.isnan(self.utility_matrix[anime_id, user_id]) and self.id_to_idx.get(anime_id) is not None:
                rating = self.predict(user_id, anime_id)
                predictions[anime_id] = rating
        print('用户推荐结束')
        recommend_animes = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:20]
        print(f'Content-Based {"MinHash" if not self.minhash else "基础"}版对用户{user_id}推荐如下动漫：')
        print('Anime\t\tScore')
        print('--------------------------')
        for anime in recommend_animes:
            print(f'{anime[0]:5d}\t\t{anime[1]:.3f}')
        return recommend_animes