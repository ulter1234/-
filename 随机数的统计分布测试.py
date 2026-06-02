import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------------------------------------------------------
# 1. 模拟 C 语言中经典的 LCG (线性同余生成器)
# ---------------------------------------------------------
class C_Rand_Simulator:
    def __init__(self, seed=2026):
        self.state = seed
        self.a = 214013
        self.c = 2531011
        self.m = 2**31

    def rand(self):
        self.state = (self.a * self.state + self.c) % self.m
        return (self.state >> 16) & 0x7FFF

# ---------------------------------------------------------
# 2. 核心统计逻辑 (完全覆盖 0 到 N-1)
# ---------------------------------------------------------
def run_5_tests(N=10, sample_size=100000):
    rng = C_Rand_Simulator()
    seq = np.array([rng.rand() % N for _ in range(sample_size)])
    results = {}

    # 1. 频数测试 (Frequency Test)
    counts = np.bincount(seq, minlength=N)
    probabilities = counts / sample_size
    expected_count = sample_size / N
    chi2_freq = np.sum((counts - expected_count)**2 / expected_count)
    results['1_频数测试卡方值'] = round(chi2_freq, 4)
    
    # 2. 间隙测试 (Gap Test) - 统计所有数字的间隔！
    all_gaps = []      # 用于画图，保存每个数字的完整间隔列表
    mean_gaps = []     # 仅仅保存每个数字的平均间隔，用于打印
    
    for i in range(N):
        idx_i = np.where(seq == i)[0]    # 找到数字 i 出现的所有索引
        gaps_i = np.diff(idx_i)          # 计算相邻索引的差值（即间隔）
        all_gaps.append(gaps_i)
        mean_gaps.append(np.mean(gaps_i) if len(gaps_i) > 0 else 0)

    # 打印时展示 0 到 N-1 所有数字的平均间隔
    results['2_所有数字的平均出现间隔'] = [round(m, 2) for m in mean_gaps]
    results['   (理论期望间隔)'] = N

    # 3. 游程测试 (Runs Test)
    median = (N - 1) / 2
    binary_seq = (seq > median).astype(int)
    # 一行代码快速统计游程数（NumPy 矢量化操作更高效）
    runs = 1 + np.sum(binary_seq[1:] != binary_seq[:-1]) 
    expected_runs = 2 * sample_size * 0.5 * 0.5 + 1 
    results['3_实际游程总数'] = runs
    results['   (理论游程期望)'] = int(expected_runs)

    # 4. 扑克测试 (Poker / Serial Test)
    pairs = seq[:-1] * N + seq[1:] 
    pair_counts = np.bincount(pairs, minlength=N**2)
    expected_pairs = (sample_size - 1) / (N**2)
    chi2_poker = np.sum((pair_counts - expected_pairs)**2 / expected_pairs)
    results['4_二维扑克测试卡方值'] = round(chi2_poker, 4)

    # 5. 块内频数测试 (Block Frequency)
    block_size = 1000
    num_blocks = sample_size // block_size
    chi2_blocks_sum = 0
    for i in range(num_blocks):
        block = seq[i*block_size : (i+1)*block_size]
        b_counts = np.bincount(block, minlength=N)
        b_expected = block_size / N
        chi2_blocks_sum += np.sum((b_counts - b_expected)**2 / b_expected)
    results['5_块内频数平均卡方值'] = round(chi2_blocks_sum / num_blocks, 4)

    return seq, probabilities, all_gaps, results

# ---------------------------------------------------------
# 3. 执行测评并直接生成图片
# ---------------------------------------------------------
if __name__ == "__main__":
    N_value = 10
    K_samples = 100000
    
    print(f"--- 正在对 rand() % {N_value} 进行 {K_samples} 次样本的 5 项测评 ---\n")
    seq, probs, all_gaps, test_results = run_5_tests(N=N_value, sample_size=K_samples)
    
    # 打印终端数据
    for key, value in test_results.items():
        print(f"{key}: {value}")
        
    print("\n--- 测试完成，正在生成分布图表图片 ---")
    
    # 画图设置
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 图 1：概率分布柱状图
    x_ticks = np.arange(N_value)
    axes[0].bar(x_ticks, probs, color='skyblue', edgecolor='black')
    axes[0].axhline(y=1/N_value, color='red', linestyle='--', label=f'理论均匀概率 ({1/N_value:.4f})')
    axes[0].set_title(f'rand() % {N_value} 的概率分布 (样本量: {K_samples})')
    axes[0].set_xlabel('数值 [0, N-1]')
    axes[0].set_ylabel('出现概率')
    axes[0].set_xticks(x_ticks)
    axes[0].legend()
    
    # 图 2：所有数字的间隔分布（箱线图 Boxplot）
    # 使用箱线图可以完美展示所有数字的分布情况（中位数、四分位数和异常值）
    axes[1].boxplot(all_gaps, labels=[str(i) for i in range(N_value)], patch_artist=True,
                    boxprops=dict(facecolor='lightgreen', color='black'),
                    medianprops=dict(color='red', linewidth=2))
    axes[1].set_title('所有数值的重复间隔分布 (箱线图 Boxplot)')
    axes[1].set_xlabel('数值 [0, N-1]')
    axes[1].set_ylabel('间隔长度 (次)')
    
    # 直接保存为图片，拒绝一切弹窗烦恼！
    plt.tight_layout()
    plt.savefig("测评分布图表_完整版.png", dpi=300, bbox_inches='tight')
    print("图表已成功保存为当前目录下的 '测评分布图表_完整版.png'！")
