import warnings

# 过滤不必要的警告
warnings.filterwarnings("ignore", message=".*weights_only=False.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn.functional as F
import numpy as np
import copy
import gc
import sys
import csv
import os
import time
from datetime import datetime
from tqdm import tqdm
from torch_geometric.loader import NeighborLoader

import shap

# 导入自定义模块
from data.config import get_args
from data.dataset import load_data
from data.models import get_model
from data.utils import set_seed, atrc

# 导入核心组件
from robust_scorer import RobustScorer
from ranker_xgboost import XGBRanker


def get_sampling_config(args, model_type):
    if model_type in ["GCN", "SAGE", "ClusterGCN", "EnGCN"]:
        return [15, 10]
    elif model_type == "SIGN":
        return [10, 10, 10]
    elif model_type == "GAT":
        return [10, 5]
    else:
        return [10, 10]


def train_gnn(args, model, data):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()

    use_loader = data.num_nodes > 50000

    if use_loader:
        num_neighbors = get_sampling_config(args, args.model_type)
        train_loader = NeighborLoader(data.cpu(), num_neighbors=num_neighbors, batch_size=args.batch_size,
                                      input_nodes=data.train_mask.cpu(), shuffle=True, num_workers=0)

    for epoch in range(args.epochs):
        if use_loader:
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False):
                batch = batch.to(args.device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index)
                loss = F.cross_entropy(out[:batch.batch_size], batch.y[:batch.batch_size])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        else:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def evaluate_gnn(model, data, args):
    model.eval()
    use_loader = data.num_nodes > 50000
    if use_loader:
        data_cpu = data.cpu()
        inference_loader = NeighborLoader(data_cpu, num_neighbors=[-1], batch_size=2048, input_nodes=None,
                                          shuffle=False, num_workers=0)
        preds_list, logits_list = [], []
        for batch in tqdm(inference_loader, desc="Inference", leave=False):
            batch = batch.to(args.device)
            out = model(batch.x, batch.edge_index)
            logits_list.append(out[:batch.batch_size].cpu())
            preds_list.append(out[:batch.batch_size].argmax(dim=1).cpu())
        logits = torch.cat(logits_list, dim=0)[:data.num_nodes]
        preds = torch.cat(preds_list, dim=0)[:data.num_nodes]
    else:
        out = model(data.x, data.edge_index)
        logits = out.cpu()
        preds = out.argmax(dim=1).cpu()
    y_true = data.y.cpu()
    is_error = (preds != y_true).float().numpy()
    if hasattr(data, 'test_mask'):
        acc = (preds[data.test_mask.cpu()] == y_true[data.test_mask.cpu()]).float().mean()
    else:
        acc = 0.0
    return logits, preds, is_error, acc.item()


def save_result_to_csv(dataset, model, acc, atrc, args, filename="experiment_results.csv"):
    file_exists = os.path.isfile(filename)
    headers = ['Timestamp', 'Dataset', 'Model', 'GNN_Acc', 'ATRC_Score', 'Total_Failures', 'LR', 'Seed', 'Config']
    seed_val = getattr(args, 'seed', 'Fixed_42')
    try:
        with open(filename, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists: writer.writerow(headers)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                dataset, model, f"{acc:.4f}", f"{atrc:.4f}",
                getattr(args, '_temp_failures', "N/A"), args.lr, seed_val,
                "X-RobustRank_WhiteBox_Explainable"  # 更新实验配置名称
            ])
        print(f"   >>> Result saved to {filename}")
    except Exception as e:
        print(f"   [Error] CSV Save failed: {e}")


def explain_selections(ranker, X_selected, feature_names, top_k=3):
    """
    [选题5 核心] 生成可解释性故障诊断报告
    使用 SHAP 值解释为什么模型认为这些节点是高风险的。
    """
    if shap is None or len(ranker.models) == 0:
        return

    print(f"\n{'=' * 60}")
    print(f"   X-RobustRank Diagnostic Report (Top-{top_k} Risks)")
    print(f"{'=' * 60}")

    # 选取 Bagging 中的一个基模型进行解释 (为了速度)
    # 或者可以对所有模型解释取平均，这里演示单模型
    model = ranker.models[0]

    # 使用 TreeExplainer 解释 XGBoost
    try:
        explainer = shap.TreeExplainer(model)
        # 计算 SHAP 值
        shap_values = explainer.shap_values(X_selected[:top_k])

        for i in range(min(top_k, len(X_selected))):
            pred_prob = model.predict_proba(X_selected[i:i + 1])[:, 1][0]
            print(f"\n[Rank #{i + 1}] Failure Probability: {pred_prob:.4f}")
            print("   >>> Risk Contributors (Why select this node?):")

            # 获取该样本的 SHAP 值
            vals = shap_values[i]

            # 找到绝对值最大的前 3 个特征
            top_indices = np.argsort(np.abs(vals))[::-1][:4]

            for idx in top_indices:
                feat_name = feature_names[idx]
                contribution = vals[idx]
                feat_value = X_selected[i][idx]

                # 解释逻辑
                direction = "INCREASES risk (+)" if contribution > 0 else "DECREASES risk (-)"
                print(f"       * {feat_name:<20} = {feat_value:.4f}  | Effect: {direction} ({contribution:.4f})")

                # [选题4 联动] 如果是梯度敏感度，特别标注
                if "Grad_Sens" in feat_name and contribution > 0:
                    print(f"         [White-Box Alert] High gradient sensitivity detected! Model is fragile here.")
                if "Neighbor_Disagree" in feat_name and contribution > 0:
                    print(f"         [Context Alert] Conflict with neighbors detected!")

    except Exception as e:
        print(f"   [Error] SHAP explanation failed: {e}")
    print(f"{'=' * 60}\n")


def generate_feature_names():
    """
    生成 38 维特征的名称列表，与 ranker_xgboost.py 中的构造逻辑对应。
    """
    # 1. 基础特征 (12维)
    base_names = [
        "Uncertainty(Drop)", "Grad_Sens(WhiteBox)", "Struct_Sens", "OOD_Dist",
        "Neighbor_Disagree", "Model_Bias", "Homophily", "MLP_Entropy",
        "Gini_Impurity", "Margin", "Entropy", "Max_Prob"
    ]

    # 2. 上下文特征 (12维)
    context_names = [f"Ctx_{name}" for name in base_names]

    # 3. 差值特征 (12维)
    diff_names = [f"Diff_{name}" for name in base_names]

    # 4. 其他 (2维)
    other_names = ["Pred_Homophily", "Log_Degree"]

    return base_names + context_names + diff_names + other_names


def run_experiment(args, dataset_name, model_name):
    current_args = copy.deepcopy(args)
    current_args.dataset = dataset_name
    current_args.model_type = model_name
    print(f"\n{'=' * 60}\nExperiment: [{dataset_name}] + [{model_name}]\n{'=' * 60}")

    try:
        dataset, data = load_data(current_args)
        if current_args.device == 'cpu':
            data = data.to('cpu')
        elif data.num_nodes <= 50000:
            data = data.to(current_args.device)
        model = get_model(current_args, dataset.num_features, dataset.num_classes).to(current_args.device)
    except Exception as e:
        print(f"Error initializing: {e}")
        return 0.0

    # 2. Step 1: 训练
    print("Step 1: Training GNN Backbone...")
    try:
        model = train_gnn(current_args, model, data)
        logits, preds, is_error, test_acc = evaluate_gnn(model, data, current_args)
        print(f"   -> GNN Test Accuracy: {test_acc:.4f}")
    except Exception as e:
        print(f"   [Error] Training failed: {e}")
        return 0.0

    # 3. Step 2: 稳健特征计算 (含白盒梯度信息)
    print("Step 2: Calculating Robust Scores (inc. White-box Gradients)...")
    scorer = RobustScorer(current_args, dataset.num_features, dataset.num_classes, current_args.device)
    if data.num_nodes > 50000:
        scorer.train_aux_mlp(data.x.cpu(), data.y.cpu(), data.train_mask.cpu())
        x_for_mlp = data.x.cpu()
    else:
        scorer.train_aux_mlp(data.x, data.y, data.train_mask)
        x_for_mlp = data.x
    mlp_logits = scorer.get_mlp_logits(x_for_mlp)

    # 这一步计算了包含 Grad_Sens 在内的8个特征
    robust_scores = scorer.get_scores(model, data, preds, mlp_logits=mlp_logits)

    # 4. Step 3: XGBoost 排序
    print(f"Step 3: XGBoost Ensemble Ranking...")
    ranker = XGBRanker(current_args)

    # 构造精简特征 (38维)
    all_features = ranker.construct_features(logits, mlp_logits, robust_scores, data.edge_index.cpu(), data.num_nodes)

    # 生成特征名称用于解释
    feature_names = generate_feature_names()

    val_mask = data.val_mask.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()

    xgb_train_X = all_features[val_mask]
    xgb_train_y = is_error[val_mask]

    xgb_test_X = all_features[test_mask]
    xgb_test_y_gt = is_error[test_mask]

    total_failures = int(xgb_test_y_gt.sum())
    current_args._temp_failures = total_failures
    print(f"   Test Set Size: {len(xgb_test_y_gt)}, Total Failures: {total_failures}")

    if total_failures == 0:
        save_result_to_csv(dataset_name, model_name, test_acc, 0.0, current_args)
        return 0.0

    # --- 迭代式排序 ---
    final_ranked_indices_local = []

    current_X_train = xgb_train_X
    current_y_train = xgb_train_y
    current_sample_weights = np.ones(len(current_y_train), dtype=np.float32)

    remaining_mask_local = np.ones(len(xgb_test_y_gt), dtype=bool)

    if total_failures < 20:
        num_rounds = 2
    elif total_failures < 100:
        num_rounds = 5
    else:
        num_rounds = 10

    target_budget = total_failures
    step_size = max(5, int(target_budget / num_rounds))

    print(f"   Iterative Ranking ({num_rounds} rounds, step_size={step_size})...")

    for r in range(num_rounds):
        ranker.fit_ensemble(current_X_train, current_y_train, sample_weights=current_sample_weights)

        local_remaining_idx = np.where(remaining_mask_local)[0]
        if len(local_remaining_idx) == 0: break

        X_remaining = xgb_test_X[local_remaining_idx]
        scores = ranker.predict_ensemble(X_remaining)

        k = min(step_size, len(scores))
        top_k_rel = np.argsort(scores)[::-1][:k]
        selected_local_idx = local_remaining_idx[top_k_rel]

        final_ranked_indices_local.extend(selected_local_idx)

        new_X = xgb_test_X[selected_local_idx]
        new_y = xgb_test_y_gt[selected_local_idx]

        current_X_train = np.vstack([current_X_train, new_X])
        current_y_train = np.concatenate([current_y_train, new_y])

        round_multiplier = 1.0 + (r * 0.2)
        new_weights = np.ones(len(new_y), dtype=np.float32)
        new_weights[new_y == 1] = 3.0 * round_multiplier
        new_weights[new_y == 0] = 1.5 * round_multiplier

        current_sample_weights = np.concatenate([current_sample_weights, new_weights])
        remaining_mask_local[selected_local_idx] = False

    if remaining_mask_local.sum() > 0:
        local_remaining_idx = np.where(remaining_mask_local)[0]
        X_remaining = xgb_test_X[local_remaining_idx]
        scores = ranker.predict_ensemble(X_remaining)
        rest_rel = np.argsort(scores)[::-1]
        final_ranked_indices_local.extend(local_remaining_idx[rest_rel])

    # 结果计算
    final_ranked_indices_local = np.array(final_ranked_indices_local)
    atrc_score = atrc(xgb_test_y_gt.astype(bool), final_ranked_indices_local, total_failures, verbose=True)

    # --- [选题5 核心] 生成可解释性报告 ---
    # 选取排名最高的 3 个节点进行深入分析
    # 这些节点是模型认为最危险的，分析它们能展示白盒特征的价值
    if len(final_ranked_indices_local) > 0:
        top_risk_features = xgb_test_X[final_ranked_indices_local[:5]]
        explain_selections(ranker, top_risk_features, feature_names, top_k=3)

    print(f">>> Result: ATRC = {atrc_score:.4f}")
    save_result_to_csv(dataset_name, model_name, test_acc, atrc_score, current_args)

    del model, data, scorer, ranker, all_features
    gc.collect()
    torch.cuda.empty_cache()

    return atrc_score


def main():
    args = get_args()

    current_seed = 42
    print(f"\n>>> [System] Using Fixed Seed: {current_seed}")
    set_seed(current_seed)
    try:
        args.seed = current_seed
    except:
        pass

    if args.run_all_datasets:
        target_datasets = args.all_datasets
    else:
        target_datasets = [args.dataset]

    if args.run_all_models:
        target_models = args.all_models
    else:
        target_models = [args.model_type]

    for ds in target_datasets:
        for md in target_models:
            run_experiment(args, ds, md)


if __name__ == "__main__":
    main()