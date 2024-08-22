import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_excel('D1_check_18.xlsx', sheet_name='Positions')

positions = df[['X', 'Y']].values

delta = 1.5 
distance_threshold = 1
total_clustered_particles = 0

# 计算任意两粒子之间的距离
def calculate_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

# 判断两粒子是否结合形成簇
def check_cluster(p1, p2):
    distance = calculate_distance(p1, p2)
    d = 10
    return distance - d - 2*delta <= distance_threshold

clusters = []

# 遍历每个粒子
for i in range(len(positions)):
    p1 = positions[i]
    cluster_found = False

    # 找到粒子所属的簇
    belonging_clusters = []
    for cluster in clusters:
        for j in cluster:
            p2 = positions[j]
            if check_cluster(p1, p2):
                belonging_clusters.append(cluster)

    # 如果粒子所属的簇数大于1，则将这些簇合并成一个更大的簇
    if len(belonging_clusters) > 1:
        new_cluster = []
        for cluster in belonging_clusters:
            if cluster in clusters:
                new_cluster.extend(cluster)
                clusters.remove(cluster)
        new_cluster.append(i)
        clusters.append(new_cluster)
        cluster_found = True

    # 如果粒子只属于一个簇，则将其添加到该簇中
    elif len(belonging_clusters) == 1:
        belonging_clusters[0].append(i)
        cluster_found = True

    #如果粒子不属于任何簇，则创建新簇
    if not cluster_found:
        clusters.append([i])
#只统计cluster中粒子数大于1的cluster        
clusters = [cluster for cluster in clusters if len(cluster) >= 2]
# 计算成簇粒子的数量N1
N1 = 0
for cluster in clusters:
    N1 += len(cluster)

    
N2 = 0
computed_pairs = set()  # 用于存储已计算过的粒子对的索引

for cluster in clusters:
    particles_count = len(cluster)
    count = 0
    for i in range(len(cluster)):
        p1 = positions[cluster[i]]
        p2 = positions[cluster[(i + 1) % particles_count]]  # 调整 i+1 的值来处理最后一个粒子

        angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        angle_degrees = math.degrees(angle)

        # 计算位置向量与y轴的夹角
        angle_y = math.degrees(math.atan2(p2[0] - p1[0], p2[1] - p1[1]))

        if abs(angle_y) < 15:
            # 创建有序的粒子对索引
            pair = (min(cluster[i], cluster[(i + 1) % particles_count]), max(cluster[i], cluster[(i + 1) % particles_count]))
            if pair not in computed_pairs:  # 检查是否已计算过这个粒子对
                count += 1
                computed_pairs.add(pair)  # 将粒子对索引添加到已计算集合中

    N2 += count


print(N1)
    
print(N2)
print(N2/N1)
print(N1/625)
