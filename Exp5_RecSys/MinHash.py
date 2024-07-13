import numpy as np
import pandas as pd

class MinHash:
    # 计算MinHash相似度
    @staticmethod
    def cal_similarity(signature_matrix, idx: int) -> np.ndarray:
        tmp_mat = signature_matrix - signature_matrix[:, idx].reshape(-1, 1)
        res_mat = 1 - np.count_nonzero(tmp_mat, axis=0) / tmp_mat.shape[0]
        res_mat[res_mat == 1] = 0
        return res_mat

    # 计算MinHash相似矩阵
    @staticmethod
    def build_similarity_matrix(signature_matrix) -> np.ndarray:
        """计算 MinHash 相似矩阵"""
        n = signature_matrix.shape[1]
        similarity_matrix = np.zeros((n, n))
        for i in range(n):
            similarity_matrix[i] = MinHash.cal_similarity(signature_matrix,i)
        similarity_matrix = pd.DataFrame(similarity_matrix, index=range(1, similarity_matrix.shape[1] + 1), columns=range(1, similarity_matrix.shape[1] + 1)).astype(float)
        return similarity_matrix.to_numpy()

class ContentBasedMinHash(MinHash):
    # 通过效用矩阵生成签名矩阵
    @staticmethod
    def build_signature_matrix(utility_matrix, hash_func_nums=10):
        array_utility_matrix = np.array(utility_matrix).T
        feature_matrix = np.where(array_utility_matrix != 0, 1, 0)
        signature_mat = np.zeros(
            (hash_func_nums, array_utility_matrix.shape[1]))
        # 随机数种子列表，用于固定哈希函数
        seed_array = np.arange(hash_func_nums)
        for i in range(hash_func_nums):
            np.random.seed(seed_array[i])
            np.random.shuffle(feature_matrix)
            row = []
            for j in range(feature_matrix.shape[1]):
                try:
                    row.append(np.where(feature_matrix[:, j] == 1)[0][0])
                except:
                    row.append(float('inf'))
            signature_mat[i] = np.array(row)
        return signature_mat

class CollaborativeFilteringMinHash(MinHash):
    # 通过效用矩阵生成签名矩阵
    @staticmethod
    def build_signature_matrix(utility_matrix, hash_func_nums=10):
        array_utility_matrix = np.array(utility_matrix)
        feature_matrix = np.where(array_utility_matrix > 5, 1, 0)
        signature_mat = np.zeros(
            (hash_func_nums, array_utility_matrix.shape[1]))
        # 随机数种子列表，用于固定哈希函数
        seed_array = np.arange(hash_func_nums)
        for i in range(hash_func_nums):
            np.random.seed(seed_array[i])
            np.random.shuffle(feature_matrix)
            row = []
            for j in range(feature_matrix.shape[1]):
                try:
                    row.append(np.where(feature_matrix[:, j] == 1)[0][0])
                except:
                    row.append(float('inf'))
            signature_mat[i] = np.array(row)
        return signature_mat