import pandas as pd
from KMeans import KMeans

def load_dataset(file='kmeans.csv'):
    dataset = pd.read_csv(file)
    dataset = dataset.to_numpy()
    return dataset

if __name__ == '__main__':
    dataset = load_dataset()
    kmeans = KMeans(dataset, 3, True)
    kmeans.run()
