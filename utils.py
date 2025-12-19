import numpy as np
import torch
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def atrc(all_failure_mask, select_test_idx, failure_num, verbose=True):
    """
    GraphRank 论文定义的 ATRC 计算逻辑 (Range: 0.1*TF to 1.0*TF)
    计算 Budget 从 0.1 * TF 到 1.0 * TF 的平均 TRC。

    :param all_failure_mask: 真实错误标签向量 (Ground Truth)
    :param select_test_idx:  全量排序后的节点索引 (sorted_indices)
    :param failure_num:      总错误数 (Total Failures, TF)
    :param verbose:          是否打印详细的 TRC 节点 (10%, 20%...)
    """
    # 1. 格式转换与基础检查
    if isinstance(select_test_idx, torch.Tensor):
        select_test_idx = select_test_idx.cpu().numpy()
    if isinstance(all_failure_mask, torch.Tensor):
        all_failure_mask = all_failure_mask.cpu().numpy()

    if failure_num == 0:
        return 0.0

    # 2. 确定 Budget 范围 [0.1 * TF, 1.0 * TF]
    # start_budget 至少为 1
    start_budget = int(0.1 * failure_num)
    if start_budget < 1:
        start_budget = 1
    end_budget = failure_num

    # 3. 预计算累积检出数 (Cumulative Failures)
    # 只需要计算到 end_budget 即可，但 select_test_idx 可能比 TF 长或短
    # 为了安全，截取 effective length
    effective_len = min(len(select_test_idx), len(all_failure_mask))

    # 获取排序后的标签 (1=错误, 0=正确)
    sorted_labels = all_failure_mask[select_test_idx[:effective_len]]

    # 计算累积和: cum_hits[k] 表示前 k+1 个样本中包含的错误数
    cum_hits = np.cumsum(sorted_labels)

    # 4. 计算指定范围内的 TRC
    # budgets 数组: [start, start+1, ..., end]
    # 注意边界：如果 effective_len 小于 end_budget，只能算到 effective_len
    real_end = min(end_budget, effective_len)
    if real_end < start_budget:
        return 0.0  # 无法计算

    budgets = np.arange(start_budget, real_end + 1)

    # 对应的检出数 (注意数组索引是 budget-1)
    hits_at_budgets = cum_hits[budgets - 1]

    # TRC = Detected / Ideal
    # 在 range [0.1TF, 1.0TF] 内，Ideal Detectable = Budget (因为 Budget <= TF)
    trc_values = hits_at_budgets / budgets

    # 5. 打印关键节点的 TRC (10%, 20% ... 100%)
    if verbose:
        print(f"\n   [ATRC Detail] Total Failures: {failure_num}")
        print("   " + "-" * 65)
        print("   | Budget Ratio |  Budget  | Detected |    TRC    |")
        print("   " + "-" * 65)

        ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for r in ratios:
            b = int(r * failure_num)
            if b < 1: b = 1
            if b > effective_len:
                print(f"   |     {r:.1f}x TF   |  {b:6d}  |   N/A    |    N/A    |")
                continue

            hits = cum_hits[b - 1]
            trc_val = hits / b
            print(f"   |     {r:.1f}x TF   |  {b:6d}  |  {hits:6d}  |   {trc_val:.4f}  |")
        print("   " + "-" * 65 + "\n")

    # 6. 计算最终 ATRC (区间平均值)
    final_atrc = np.mean(trc_values)

    return final_atrc