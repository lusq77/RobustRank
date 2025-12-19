import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description="RobustRank: Large Scale & Scalable")

    # --- 数据集控制 ---
    parser.add_argument("--dataset", type=str, default="Cora",
                        help="Small: Cora, CiteSeer, PubMed; Large: Reddit, Flickr, ogbn-products")
    parser.add_argument("--root", type=str, default="./data")

    parser.add_argument("--run_all_datasets", action="store_true", help="Run all datasets sequentially")
    parser.add_argument("--all_datasets", type=str, nargs='+', default=["Cora", "CiteSeer", "PubMed"])

    # --- 模型控制 ---
    parser.add_argument("--model_type", type=str, default="GCN",
                        choices=["GCN", "GAT", "TAGCN", "SAGE", "SIGN", "EnGCN", "ClusterGCN"])
    parser.add_argument("--run_all_models", action="store_true", help="Run all models sequentially")
    parser.add_argument("--all_models", type=str, nargs='+', default=["GCN", "GAT", "TAGCN"])

    # --- 训练超参 ---
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=10000)

    # 特定模型参数
    parser.add_argument("--gat_heads", type=int, default=4)
    parser.add_argument("--tagcn_k", type=int, default=2)
    parser.add_argument("--sign_hops", type=int, default=3)

    # --- RobustRank 参数 ---
    parser.add_argument("--selection_budget", type=int, default=50)
    parser.add_argument("--ablation", type=str, default="none")
    parser.add_argument("--fgsm_epsilon", type=float, default=0.01)

    # XGBoost
    parser.add_argument("--xgb_estimators", type=int, default=100)
    parser.add_argument("--xgb_depth", type=int, default=4)
    parser.add_argument("--xgb_lr", type=float, default=0.1)

    # --- 【修复】新增随机种子参数 ---
    # 默认为 None，表示不固定种子（将由 main.py 生成时间戳种子）
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: None, using timestamp)")

    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device)

    args = parser.parse_args()
    return args