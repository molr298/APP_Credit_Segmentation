import pandas as pd
import numpy as np

df = pd.read_csv('../CC GENERAL.csv')

print(df.head())
print("======================")
print(df.isnull().sum())
print("======================")

df['MINIMUM_PAYMENTS'] = df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median())
df['CREDIT_LIMIT'] = df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].mean())
print(df.isnull().sum())
print("======================")

df = df.drop('CUST_ID', axis=1)
np_data = df.to_numpy()
print(np_data.shape)
print("======================")

k = 7
cluster = np.zeros(np_data.shape[0])
centroid = np_data[np.random.randint(np_data.shape[0], size=k), :]


def calc_distance(data, data_centroid):
    dist = np.zeros((data.shape[0], data_centroid.shape[0]))
    for i in range(data.shape[0]):
        for j in range(data_centroid.shape[0]):
            dist[i][j] = np.linalg.norm(data[i] - centroid[j])
    return dist


def get_nearest_centroid(distance):
    return np.argmin(distance, axis=1)


def get_new_centroids(data, data_nearest_centroid, number_of_centroid):
    result_centroids = np.zeros((number_of_centroid, data.shape[1]))
    for i in range(number_of_centroid):
        result_centroids[i] = data[np.where(data_nearest_centroid == i)].mean(axis=0)
    return result_centroids


# calculated_dist = calc_distance(np_data, centroid)
# print(calculated_dist.shape)  # (8950, 3)
#
# nearest_centroid = get_nearest_centroid(calculated_dist)
# print(nearest_centroid.shape)  # 8950
#
# new_centroids = get_new_centroids(np_data, nearest_centroid, k)
# print(new_centroids.shape)  # (3, 17)

iteration = 0
has_changed_centroid = True
while has_changed_centroid:
    print(f"iter {iteration}")
    calculated_dist = calc_distance(np_data, centroid)  # calculated dist
    nearest_centroid = get_nearest_centroid(calculated_dist)  # assigned to centroid
    new_centroid = get_new_centroids(np_data, nearest_centroid, k)
    if np.all(new_centroid == centroid):
        has_changed_centroid = False
    else:
        print(f"changed {np.linalg.norm(centroid - new_centroid)}")
        centroid = new_centroid
        iteration += 1
print("new centroids:")
print(centroid)
print("iteration took:")
print(iteration)
