import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd


def generate_particles(N):
    # 计算粒子间隔
    c = L / np.sqrt(N)
    # 在x和y轴上生成等间距的点阵
    x, y = np.meshgrid(np.arange(0, L, c), np.arange(0, L, c))
    # 将两个矩阵合并成一个形状为(N, 2)的数组
    positions = np.vstack((x.ravel(), y.ravel())).T
    # 添加随机扰动,loc=1/10c,scale=1/20c
    positions += np.random.normal(loc=c / 10, scale=c / 20, size=(N, 2))
    theta = np.random.uniform(low=0, high=2 * np.pi, size=N)  # 随机生成N个角度
    m = np.vstack((np.cos(theta), np.sin(theta))).T  # 转化为磁矩方向单位向量,产生(N，2)的数组
    return positions, m


# 计算外加磁场作用势
@nb.njit(fastmath=True)
def calculate_U_hi(m_i):
    lambda_h = 2.19e-20  # 1000Gs
    h = np.array([0, 1])
    return -lambda_h * (m_i[0] * h[0] + m_i[1] * h[1])


# 计算磁偶极子相互作用势
@nb.njit(fastmath=True)
def calculate_U_mij(m_i, m_j, r_ij):
    lambda_m = 4.7524e-21  # 磁偶极子作用势强度
    r_ij_norm = np.linalg.norm(r_ij)
    t_ij = r_ij / r_ij_norm
    # term1 = np.dot(m_i, m_j)
    term1 = np.dot(np.ascontiguousarray(m_i), np.ascontiguousarray(m_j))
    term2 = 3 * np.dot(m_i, t_ij) * np.dot(m_j, t_ij)
    # d_p < cutoff_U_m < 10*d_p
    if d_p < r_ij_norm <= 10 * d_p:
        return lambda_m * ((d_p / r_ij_norm) ** 3) * (term1 - term2)
    else:
        return 0


# 计算活性剂排斥势
@nb.njit(fastmath=True)
def calculate_U_vij(r_ij):
    lambda_v = 6.5e-19  # 活性剂排斥势强度
    term1 = 2 + (np.linalg.norm(r_ij) / delta) * np.log(np.linalg.norm(r_ij) / d)
    term2 = (np.linalg.norm(r_ij) - d_p) / delta
    # d_p < cutoff_U_v <= d
    if d_p < np.linalg.norm(r_ij) <= d:
        return lambda_v * (term1 - term2)
    else:
        return 0


# 计算总势能
@nb.njit(fastmath=True)
def calculate_energy(positions, m):
    U_h = 0  # 外加磁场作用势
    U_m = 0  # 磁偶极子相互作用势
    U_v = 0  # 表面活性剂的排斥势
    N = len(positions)
    for i in nb.prange(N):
        U_h += calculate_U_hi(m[i])
        for j in range(i + 1, N):
            r_ij = positions[i] - positions[j]
            U_m += calculate_U_mij(m[i], m[j], r_ij)
            U_v += calculate_U_vij(r_ij)

    energy_total = U_v + U_m + U_h
    return energy_total, U_h, U_m, U_v


# Metropolis采样
def metropolis(positions, m):
    accepted = 0
    kb = 1.38e-23  # J/K
    T = 300  # 温度 K
    beta = 1 / (kb * T)  # 计算beta值
    energy_list = []  # 记录能量变化
    U_h_list = []
    U_m_list = []
    U_v_list = []
    acceptance_list = []
    N = len(positions)
    fig = 0
    with tqdm(total=N_steps, ascii=True) as pbar:
        for step in range(N_steps):
            # 每次循环开始前更新进度条
            pbar.update(1)
            # 计算当前状态的能量
            energy_i, U_h, U_m, U_v = calculate_energy(positions, m)
            # 随机选取一个粒子,最大位移为0.5d
            alpha = np.random.randint(N)
            a = 0.5 * d
            gamma = np.random.random()
            phi = np.random.uniform(0, 2 * np.pi)
            delta_position = np.array([a * gamma * np.cos(phi), a * gamma * np.sin(phi)])

            # 计算新状态的位置，考虑周期性边界条件
            new_positions = positions.copy()
            for k in range(2):
                new_positions[alpha][k] += delta_position[k]
                if new_positions[alpha][k] < 0:
                    new_positions[alpha][k] += L
                elif new_positions[alpha][k] > L:
                    new_positions[alpha][k] -= L
            new_positions[alpha] += delta_position


            "碰撞检测"
            has_collision = False
            for i in range(len(positions)):
                if i != alpha:
                    if np.linalg.norm(new_positions[alpha] - new_positions[i]) < d:
                        has_collision = True
                        break
            if not has_collision:
                # 计算新状态的能量
                energy_j, _, _, _ = calculate_energy(new_positions, m)
                # 计算能量差
                delta_E1 = energy_j - energy_i
                # 判断是否接受新状态
                if delta_E1 <= 0:
                    positions = new_positions
                    energy_i = energy_j
                    accepted += 1
                    R1 = 1
                    P1 = 1
                else:
                    P1 = np.exp(-beta * delta_E1)
                    R1 = np.random.random()
                    if R1 <= P1:
                        positions = new_positions
                        energy_i = energy_j
                        accepted += 1

            # 改变粒子alpha的磁矩方向
            new_m = m.copy()
            # 当前状态的磁矩方向
            theta = np.arctan2(m[alpha][1], m[alpha][0])
            # 改变磁矩方向
            c = np.pi / 18  # beta为磁矩与x轴正方向的最大转动角度
            lamda = 2 * np.random.random() - 1  # lamda范围为[-1,1]
            delta_theta = lamda * c  # 范围为[-pi/18, pi/18]
            # 新磁矩方向
            new_theta = theta + delta_theta
            new_m[alpha][0] = np.cos(new_theta)
            new_m[alpha][1] = np.sin(new_theta)
            new_m[alpha] = new_m[alpha] / np.linalg.norm(new_m[alpha])

            # 计算新状态的能量
            energy_k, _, _, _ = calculate_energy(positions, new_m)
            # 计算能量差
            delta_E2 = energy_k - energy_i
            # 判断是否接受新状态
            if delta_E2 <= 0:
                m = new_m
                energy_i = energy_k
                accepted += 1
                R2 = 1
                P2 = 1
            else:
                P2 = np.exp(-beta * delta_E2)
                R2 = np.random.random()
                if R2 <= P2:
                    m = new_m
                    energy_i = energy_k
                    accepted += 1
            acceptance = int(R1 <= P1 and R2 <= P2)
            acceptance_list.append(acceptance)
            energy_list.append(energy_i)
            U_h_list.append(U_h)
            U_m_list.append(U_m)
            U_v_list.append(U_v)

            #20000 步保存粒子图、能量图
            if (step + 1) % 5000 == 0:
                plt.figure(figsize=(15, 15))
                plt.scatter(positions[:, 0], positions[:, 1], s=100, alpha=0.7)
                plt.xlim([0, L])
                plt.ylim([0, L])
                # 绘制磁矩方向图，以（final_positions[i,0],final_positions[i,1]）为起点
                for i in range(N):
                    plt.arrow(positions[i, 0], positions[i, 1],
                              m[i, 0], m[i, 1],
                              head_width=60, head_length=60, color='r', alpha=0.7)
                plt.title(f"step{step + 1}")
                plt.savefig(f"particles_{fig}.png")
                plt.close()
                # 绘制能量图
                plt.figure(figsize=(20, 10))
                plt.plot(range(len(energy_list)), energy_list, label='Total Energy')
                plt.plot(range(len(U_h_list)), U_h_list, label='U_h')
                plt.plot(range(len(U_m_list)), U_m_list, label='U_m')
                plt.plot(range(len(U_v_list)), U_v_list, label='U_v')
                plt.xlabel('Step number')
                plt.ylabel('Energy')
                plt.title('Energy vs. Step number')
                plt.legend()
                plt.savefig(f"energy_{fig}.png")
                plt.close()
                fig += 1
                df_m = pd.DataFrame({'m_x': m[:, 0], 'm_y': m[:, 1]})
                df_acceptance = pd.DataFrame({'Acceptance': acceptance_list})
                df_positions = pd.DataFrame({'X': positions[:, 0], 'Y': positions[:, 1]})
                df_energy = pd.DataFrame({'Energy': energy_list})
                df_U_h = pd.DataFrame({'U_h': U_h_list})
                df_U_m = pd.DataFrame({'U_m': U_m_list})
                df_U_v = pd.DataFrame({'U_v': U_v_list})

                # 保存数据到excel文件
                with pd.ExcelWriter(f'D1_check_{fig}.xlsx') as writer:
                    df_acceptance.to_excel(writer, sheet_name='Acceptance', index=False)
                    df_m.to_excel(writer, sheet_name='m', index=False)
                    df_positions.to_excel(writer, sheet_name='Positions', index=False)
                    df_energy.to_excel(writer, sheet_name='Energy', index=True)
                    df_U_h.to_excel(writer, sheet_name='U_h', index=True)
                    df_U_m.to_excel(writer, sheet_name='U_m', index=True)
                    df_U_v.to_excel(writer, sheet_name='U_v', index=True)

    return positions, m, energy_list, U_h_list, U_m_list, U_v_list, acceptance_list


# 设定参数并运行
N = 625  # 粒子数
L = 640  # nm
d_p = 10  # nm
delta = 1.5  # 活性剂厚度
d = d_p + 2 * delta
N_steps = 1000000 # 总步数

positions, m = generate_particles(N)  # 生成初始位置和磁矩向量

final_positions, final_m, energy_list, U_h_list, U_m_list, U_v_list, acceptance_list = metropolis(
    positions, m)


