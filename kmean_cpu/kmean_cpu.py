import pandas as pd
import numpy as np
import time

df = pd.read_csv('data/CC GENERAL.csv')

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

k = 20
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
total_time_start = time.perf_counter()
while has_changed_centroid:
    iter_time_start = time.perf_counter()
    calc_dist_time_start = time.perf_counter()
    calculated_dist = calc_distance(np_data, centroid)  # calculated dist
    calc_dist_time_end = time.perf_counter()
    nearest_centroid_time_start = time.perf_counter()
    nearest_centroid = get_nearest_centroid(calculated_dist)  # assigned to centroid
    nearest_centroid_time_end = time.perf_counter()
    new_centroid = get_new_centroids(np_data, nearest_centroid, k)
    if np.all(new_centroid == centroid):
        has_changed_centroid = False
    else:
        # print(f"changed {np.linalg.norm(centroid - new_centroid)}")
        centroid = new_centroid
    iter_time_end = time.perf_counter()
    print(
        f"iter {iteration} | took: {iter_time_end - iter_time_start:0.4f} |"
        f" dist: {calc_dist_time_end - calc_dist_time_start:0.4f} | "
        f"nearest: {nearest_centroid_time_end - nearest_centroid_time_start:0.4f}")
    iteration += 1
total_time_end = time.perf_counter()
# print("new centroids:")
# print(centroid)
print(f"k: {k} | iteration took: {iteration} | total time: {total_time_end - total_time_start:0.4f}")
