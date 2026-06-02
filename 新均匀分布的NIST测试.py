import numpy as np
import math
from scipy.special import erfc, gammaincc
import time

# ==========================================
# 1. 你设计的无偏差离散均匀分布公式
# ==========================================
def generate_uniform_discrete(N, sample_size):
    # 公式: X = floor(U * N)
    U = np.random.rand(sample_size)
    return np.floor(U * N).astype(int)

# ==========================================
# 2. NIST SP 800-22 核心密码学测试
# ==========================================
def nist_monobit_test(binary_seq):
    n = len(binary_seq)
    S_n = np.sum(2 * binary_seq - 1)
    s_obs = abs(S_n) / math.sqrt(n)
    return erfc(s_obs / math.sqrt(2))

def nist_block_frequency_test(binary_seq, M=128):
    n = len(binary_seq)
    N_blocks = n // M
    if N_blocks == 0: return 0.0
    blocks = binary_seq[:N_blocks * M].reshape(N_blocks, M)
    pi_i = np.sum(blocks, axis=1) / M
    chi_squared = 4 * M * np.sum((pi_i - 0.5)**2)
    return gammaincc(N_blocks / 2, chi_squared / 2)

def nist_runs_test(binary_seq):
    n = len(binary_seq)
    pi = np.sum(binary_seq) / n
    if abs(pi - 0.5) >= (2 / math.sqrt(n)):
        return 0.0
    v_n_obs = 1 + np.sum(binary_seq[:-1] != binary_seq[1:])
    numerator = abs(v_n_obs - 2 * n * pi * (1 - pi))
    denominator = 2 * math.sqrt(2 * n) * pi * (1 - pi)
    return erfc(numerator / denominator)

# ==========================================
# 3. 极速执行百万级 NIST 测试
# ==========================================
if __name__ == "__main__":
    N_value = 10000
    K_samples = 1000000  # 100万样本
    
    print(f"=======================================================")
    print(f"   新设计公式的 NIST SP 800-22 密码学测评 (N={N_value})")
    print(f"=======================================================\n")
    
    start_time = time.time()
    
    # 1. 使用新公式生成 100 万个 0~9999 的十进制随机数
    print("正在使用新公式 floor(U * 10000) 生成序列...")
    seq_new = generate_uniform_discrete(N_value, K_samples)
    
    # 2. 关键步骤：二值化转换为 NIST 要求的比特流
    # 根据中位数 5000 进行阈值划分，保证理论上的 0/1 概率严格为 50%
    print("正在进行二值化转换 (阈值映射为 0 和 1)...")
    binary_sequence = (seq_new >= (N_value / 2)).astype(int)
    
    print("\n--- 运行 NIST 核心测试项 ---")
    print(f"显著性水平 (Alpha): 0.01 (P-value >= 0.01 视为通过)\n")
    
    # 运行测试
    p1 = nist_monobit_test(binary_sequence)
    res1 = " PASS" if p1 >= 0.01 else "❌ FAIL"
    print(f"[NIST 1] 单比特频数测试 (Monobit)  : P-value = {p1:.6f} \t [{res1}]")
    
    p2 = nist_block_frequency_test(binary_sequence, M=128)
    res2 = "PASS" if p2 >= 0.01 else "❌ FAIL"
    print(f"[NIST 2] 块内频数测试 (Block Freq): P-value = {p2:.6f} \t [{res2}]")
    
    p3 = nist_runs_test(binary_sequence)
    res3 = "PASS" if p3 >= 0.01 else "❌ FAIL"
    print(f"[NIST 3] 游程测试 (Runs Test)     : P-value = {p3:.6f} \t [{res3}]")
    
    end_time = time.time()
    print("=======================================================")
    print(f" 测试总耗时: {end_time - start_time:.4f} 秒")
    print("=======================================================")