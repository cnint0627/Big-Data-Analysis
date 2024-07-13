import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
import time

from matplotlib import rcParams



class KMeans:
    def __init__(self, dataset: np.ndarray, k, random_state = True):
        self.dataset = dataset  # 数据集
        self.train = dataset[:, 1:]
        self.test = dataset[:, 0]
        self.N, self.D = self.train.shape
        self.k = k  # K个聚类
        self.k_centers = None # K个簇中心
        self.clusters = None  # K个簇所包含的样本
        self.point_to_cluster = dict()  # 各个样本所在的簇
        self.random_state = random_state    # 是否随机初始化

    # 初始化k个簇中心
    def init_k_centers(self):
        if self.random_state:
            self.k_centers = [self.train[i] for i in random.sample(range(self.N), self.k)]
            return
        k_center_1 = self.train[random.randint(0, self.N - 1)]
        k_center_1_mat = np.tile(k_center_1, (self.N, 1))
        # 寻找距离第一个簇中心最远的点作为第二个簇中心
        k_center_2 = self.train[np.argmax(np.linalg.norm(self.train - k_center_1_mat, axis=1))]
        k_center_2_mat = np.tile(k_center_2, (self.N, 1))
        # 寻找到前两个簇中心距离之和最远的点作为第三个簇中心
        k_center_3 = self.train[np.argmax(np.linalg.norm(self.train - k_center_1_mat, axis=1) + np.linalg.norm(self.train - k_center_2_mat, axis=1))]
        self.k_centers = [k_center_1, k_center_2, k_center_3]

    # 计算两点间的欧氏距离
    @staticmethod
    def calc_distance(x, y):
        return np.linalg.norm(x - y)

    # 更新各个簇的中心
    def update_k_centers(self):
        is_update = False
        for i in range(self.k):
            old_center = self.k_centers[i]
            self.k_centers[i] = np.average(self.clusters[i], axis=0)
            if not np.array_equal(old_center, self.k_centers[i]):
                is_update = True
        return is_update

    # 更新K个簇中的样本
    def update_clusters(self):
        self.clusters = [list() for _ in range(self.k)]
        for i in range(self.N):
            point = self.train[i]
            min_distance = float('inf')
            for j in range(self.k):
                distance = self.calc_distance(point, self.k_centers[j])
                if distance < min_distance:
                    min_distance = distance
                    best_k = j
            self.clusters[best_k].append(point)
            self.point_to_cluster[i] = best_k

    # 获取所有样本的标签
    def get_point_labels(self) -> list:
        labels = [self.point_to_cluster[i] + 1 for i in range(self.N)]
        return labels

    #  性能评估
    def score(self, labels: list):
        maps = itertools.permutations(range(1, self.k + 1), self.k)
        best_acc = 0
        best_labels = list()
        for map in maps:
            acc = 0
            for i in range(self.N):
                if map[labels[i] - 1] == self.test[i]:
                    acc += 1
            if acc > best_acc:
                best_acc = acc
                best_labels = [map[label - 1] for label in labels]
        for i in range(self.N):
            labels[i] = best_labels[i]

        # 计算距离平方和
        sse = 0
        for i in range(self.N):
            sse += self.calc_distance(self.train[i] ,self.k_centers[self.point_to_cluster[i]]) ** 2
        return (best_acc / self.N * 1.0, sse)

    # 将聚类结果以图的形式呈现
    def draw(self, labels: list):
        rcParams['font.family'] = 'SimHei'
        score = self.score(labels)
        fig, ax = plt.subplots()
        ax.scatter(x=self.train[:, 1], y=self.train[:, 9], c=labels)
        ax.set_xlabel("得分10", fontsize=12)
        ax.set_ylabel("得分2", fontsize=12)
        ax.set_title(f'距离平方和SSE={score[1]:.2f} 准确率Accuracy={score[0] * 100:.2f}%')
        plt.show()
    def run(self):
        start_time = time.time()
        print('KMeans starts!')
        self.init_k_centers()
        self.update_clusters()
        while self.update_k_centers():
            self.update_clusters()
        labels = self.get_point_labels()
        score = self.score(labels)
        print('KMeans ends!')
        print(f'Accuracy: {score[0] * 100:.2f}%, SSE: {score[1]:.2f}')
        print(f'Run time: {time.time() - start_time:.2f}s')
        self.draw(labels)







