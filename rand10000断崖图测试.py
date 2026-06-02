import numpy as np
import matplotlib.pyplot as plt

class C_Rand_Simulator:
    def __init__(self, seed=2026):
        self.state = seed
        self.a = 214013
        self.c = 2531011
        self.m = 2**31

    def rand(self):
        self.state = (self.a * self.state + self.c) % self.m
        return (self.state >> 16) & 0x7FFF

# --- 设定大基数和极高样本量 ---
N_value = 10000
K_samples = 1000000 

print(f"正在对 rand() % {N_value} 进行 {K_samples} 次频数测试，请稍候...")

rng = C_Rand_Simulator()
# 高速生成数组
seq = np.array([rng.rand() % N_value for _ in range(K_samples)])

# 核心频数统计：O(N) 复杂度，瞬间完成
counts = np.bincount(seq, minlength=N_value)
probs = counts / K_samples

# --- 绘图阶段 ---
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(12, 5))

# 因为 N=10000，柱状图会糊成一团，这里改用散点/区域图展示宏观分布
ax.scatter(np.arange(N_value), probs, s=1, color='teal', alpha=0.5)
ax.axhline(y=1/N_value, color='red', linestyle='--', linewidth=2, label=f'理论均匀概率 ({1/N_value:.6f})')

ax.set_title(f'模偏差暴露：rand() % {N_value} 的分布断崖 (样本量: {K_samples})')
ax.set_xlabel(f'数值区间 [0, {N_value-1}]')
ax.set_ylabel('实际出现概率')
ax.legend()

plt.tight_layout()
plt.savefig("模偏差断崖图.png", dpi=300, bbox_inches='tight')
print("瞬间完成！图表已保存为 '模偏差断崖图.png'")