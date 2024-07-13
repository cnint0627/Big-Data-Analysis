import numpy as np
import pandas as pd
from MinHash import *
from RecommendSystem import RecommendSystem

class CollaborativeFiltering(RecommendSystem):
    def __init__(self, train, minhash=False):
        super().__init__(train, minhash)
        self.similarity_matrix = None
        self.similar_users_id = None

    # 使用pearson相关系数计算用户相似度矩阵
    def build_similarity_matrix(self):
        if self.minhash:
            signature_matrix = CollaborativeFilteringMinHash.build_signature_matrix(self.utility_matrix)
            self.similarity_matrix = CollaborativeFilteringMinHash.build_similarity_matrix(signature_matrix)
        else:
            # 使用用户评分均值对用户评分进行修正
            utility_matrix_revise = self.utility_matrix - np.nanmean(self.utility_matrix, axis=0)
            # 将NaN值转换为0，便于后续点乘操作
            utility_matrix_revise = np.nan_to_num(utility_matrix_revise, nan=0)
            # 归一化修正后的效用矩阵
            utility_matrix_revise_norm = utility_matrix_revise / np.linalg.norm(utility_matrix_revise, axis=0, keepdims=True)
            # 相似度矩阵为归一化矩阵的点乘矩阵
            self.similarity_matrix = np.dot(utility_matrix_revise_norm.T, utility_matrix_revise_norm)
            # self.similarity_matrix = pd.DataFrame(self.utility_matrix).corr()

    # 构造用户-用户协同过滤模型
    def fit(self):
        print('模型构造开始')
        self.build_utility_matrix()
        self.build_similarity_matrix()
        # print(self.similarity_matrix)
        print('模型构造完毕')

    # 预测指定用户对指定动漫的评分
    def predict(self, user_id, anime_id, k) -> tuple:
        rat_k_mat = self.utility_matrix[anime_id, self.similar_users_id]
        return np.nanmean(rat_k_mat), np.count_nonzero(~np.isnan(rat_k_mat))
        # sim_k_mat = self.similarity_matrix[user_id, similar_users_id]

    # 利用k个最相似用户的评分情况对用户的未评分动漫进行推荐
    def recommend(self, user_id, k) -> list:
        # 得到k个最相近用户
        self.similar_users_id = np.argsort(-self.similarity_matrix[user_id])[1: k+1]
        print('用户推荐开始')
        predictions = dict()
        # 对用户每个未评分项目进行预测
        for anime_id in range(1, self.anime_nums+1):
            if np.isnan(self.utility_matrix[anime_id, user_id]):
                # weighted_ratings = 0
                # sims = 0
                # cnt = 0
                # for similar_user_id in similar_users_id:
                #     rating = self.utility_matrix[anime_id, similar_user_id]
                #     sim = self.similarity_matrix[user_id, similar_user_id]
                #     if not np.isnan(rating):
                #         weighted_ratings += rating * sim
                #         sims += sim
                #         cnt += 1
                rating, cnt = self.predict(user_id, anime_id, k)
                # 如果至少有一个相似用户对该动漫由评分，则对用户的评分进行预测
                if cnt >= 20:
                    predictions[anime_id] = rating
        print('用户推荐结束')
        recommend_animes = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:20]
        print(f'CollaborativeFiltering {"MinHash" if not self.minhash else "基础"}版对用户{user_id}推荐如下动漫：')
        print('Anime\t\tScore')
        print('--------------------------')
        for anime in recommend_animes:
            print(f'{anime[0]:5d}\t\t{anime[1]:.3f}')
        return recommend_animes

