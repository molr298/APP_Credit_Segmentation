import math

import pandas as pd
import numpy as np
from numba import cuda

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

k = 7
centroid = np_data[np.random.randint(np_data.shape[0], size=k), :]


def next_power_of_2(x):
    return 1 << (x - 1).bit_length()


def calc_dimension_for_distance(data, data_centroid):
    real_dim_x = data_centroid.shape[0]
    dim_x = next_power_of_2(real_dim_x)
    dim_y = dim_x
    thread_per_blocks = (dim_y, dim_x)
    blocks_per_grid_x = 1
    blocks_per_grid_y = math.ceil(data.shape[0] / thread_per_blocks[0])
    blocks_per_grid = (blocks_per_grid_y, blocks_per_grid_x)
    return thread_per_blocks, blocks_per_grid


dist_tpb, dist_bpg = calc_dimension_for_distance(np_data, centroid)


@cuda.jit
def calc_distance_kernel(data, data_centroid, result):
    r = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    c = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if r < data.shape[0] and c < data_centroid.shape[0]:
        result[r][c] = 1


def calc_distance(data, data_centroid):
    result = np.zeros((data.shape[0], data_centroid.shape[0]))
    data_device = cuda.to_device(data)
    centroid_device = cuda.to_device(centroid)
    result_device = cuda.to_device(result)

    # invoke kernel
    calc_distance_kernel[dist_bpg, dist_tpb](data_device, centroid_device, result_device)

    result = result_device.copy_to_host()
    return result


calc_distance(np_data, centroid)
