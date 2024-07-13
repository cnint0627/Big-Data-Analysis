import numpy as np

'''
两个推荐系统的父类
'''
class RecommendSystem:
    def __init__(self, train, minhash):
        self.train = train
        self.minhash = minhash
        self.user_nums = np.max(train[:, 0])
        self.anime_nums = np.max(train[:, 1])
        self.utility_matrix = None

    # 构造用户-动漫评分效用矩阵
    def build_utility_matrix(self):
        # 初始化效用矩阵，行为user，列为anime
        self.utility_matrix = np.full((self.anime_nums+1, self.user_nums+1), np.nan)
        for data in self.train:
            # data为数据集中一条用户对动漫的评分
            user_id, anime_id, rating = data
            self.utility_matrix[anime_id][user_id] = rating






