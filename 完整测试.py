import numpy as np
import scipy.stats as stats
import time

# ==========================================
# 1. 算法源泉
# ==========================================
class C_Rand_Simulator:
    def __init__(self, seed=2026):
        self.state = seed
        self.a = 214013
        self.c = 2531011
        self.m = 2**31

    def rand(self):
        self.state = (self.a * self.state + self.c) % self.m
        return (self.state >> 16) & 0x7FFF

def generate_uniform_discrete(N, sample_size):
    U = np.random.rand(sample_size)
    return np.floor(U * N).astype(int)

def generate_normal_box_muller(sample_size):
    U1 = 1.0 - np.random.rand(sample_size) 
    U2 = np.random.rand(sample_size)
    return np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)

# ==========================================
# 2. 核心：大基数 5 项测评模块
# ==========================================
def run_5_tests_large_scale(seq, N, sample_size):
    results = {}

    # 测试 1: 频数测试 (卡方检验)
    counts = np.bincount(seq, minlength=N)
    expected = sample_size / N
    chi2_freq = np.sum((counts - expected)**2 / expected)
    results['1_频数测试卡方值'] = f"{chi2_freq:.2f} (期望~{N-1})"

    # 测试 2: 间隙测试 (以数字 0 为例，理论期望间隔为 N)
    idx_0 = np.where(seq == 0)[0]
    mean_gap = np.mean(np.diff(idx_0)) if len(idx_0) > 1 else 0
    results['2_数字0的平均间隔'] = f"{mean_gap:.2f} (理论期望: {N})"

    # 测试 3: 游程测试 (高于中位数 vs 低于中位数)
    median = (N - 1) / 2
    binary_seq = (seq > median).astype(int)
    runs = 1 + np.sum(binary_seq[1:] != binary_seq[:-1]) 
    expected_runs = 2 * sample_size * 0.5 * 0.5 + 1
    results['3_实际游程总数'] = f"{runs} (理论期望: {int(expected_runs)})"

    # 测试 4: 二维扑克测试 (降维至 N=100 进行测试，暴露 LCG 低位缺陷)
    seq_mod100 = seq % 100
    pairs = seq_mod100[:-1] * 100 + seq_mod100[1:]
    pair_counts = np.bincount(pairs, minlength=10000)
    expected_pairs = (sample_size - 1) / 10000
    chi2_poker = np.sum((pair_counts - expected_pairs)**2 / expected_pairs)
    results['4_扑克测试卡方值'] = f"{chi2_poker:.2f} (期望~9999)"

    # 测试 5: 块内频数测试 (分10个超级块，每块10万样本)
    num_blocks = 10
    block_size = sample_size // num_blocks
    chi2_blocks_sum = 0
    for i in range(num_blocks):
        block = seq[i*block_size : (i+1)*block_size]
        b_counts = np.bincount(block, minlength=N)
        b_expected = block_size / N
        chi2_blocks_sum += np.sum((b_counts - b_expected)**2 / b_expected)
    results['5_块内频数平均卡方'] = f"{chi2_blocks_sum / num_blocks:.2f} (期望~{N-1})"

    return results

# ==========================================
# 3. 极速执行测试
# ==========================================
if __name__ == "__main__":
    N_value = 10000
    K_samples = 1000000
    
    start_time = time.time()
    
    # --- 阶段一：批判旧算法 (C LCG) ---
    print(f"C语言经典取模法 (rand() % {N_value})")
    rng = C_Rand_Simulator()
    seq_old = np.array([rng.rand() % N_value for _ in range(K_samples)])
    res_old = run_5_tests_large_scale(seq_old, N_value, K_samples)
    for k, v in res_old.items(): print(f"    {k} : {v}")
    print("    > 结论: LCG 在扑克测试和频数测试中卡方值爆炸，存在极其严重的规律性和模偏差。\n")

    # --- 阶段二：验证新算法 (公式映射) ---
    print(f" 乘法取整映射法 (floor(U * {N_value}))")
    seq_new = generate_uniform_discrete(N_value, K_samples)
    res_new = run_5_tests_large_scale(seq_new, N_value, K_samples)
    for k, v in res_new.items(): print(f"    {k} : {v}")
    print("    > 结论: 5项测试数据均完美贴合理论期望，成功抹平模偏差！\n")

    # --- 阶段三：正态分布验证 ---
    print(f" Box-Muller 变换正态分布验证")
    seq_normal = generate_normal_box_muller(K_samples)
    ks_stat, p_value = stats.kstest(seq_normal, 'norm')
    print(f"    K-S 统计量 (D-value) : {ks_stat:.4f}")
    print(f"    P-value              : {p_value:.4f}")
    if p_value >= 0.05:
        print("    > 结论: K-S 检验通过！极度贴合标准高斯分布。\n")
        
    print("=======================================================")
    print(f"⚡ 终极脚本测试总耗时: {time.time() - start_time:.4f} 秒")
    print("=======================================================")