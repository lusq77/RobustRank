import numpy as np
import torch
import torch.nn.functional as F
from xgboost import XGBClassifier
from torch_geometric.utils import degree
from torch_sparse import SparseTensor
import gc


class XGBRanker:
    def __init__(self, args):
        self.device = args.device
        self.models = []
        # 用于存储特征名称，供 SHAP 解释使用
        self.feature_names = []

    def aggregate_neighbors(self, features, edge_index, num_nodes):
        """聚合邻居特征均值"""
        if num_nodes > 50000:
            features = features.cpu()
            edge_index = edge_index.cpu()
            calc_device = 'cpu'
        else:
            calc_device = self.device
            if self.device != 'cpu':
                features = features.to(calc_device)
                edge_index = edge_index.to(calc_device)
        try:
            row, col = edge_index
            deg = degree(col, num_nodes, dtype=torch.float)
            deg_inv = deg.pow(-1)
            deg_inv[deg_inv == float('inf')] = 0
            adj = SparseTensor(row=row, col=col, value=deg_inv[row], sparse_sizes=(num_nodes, num_nodes))
            return adj.matmul(features).cpu()
        except:
            return features.cpu()

    def get_prediction_homophily(self, probs, edge_index, num_nodes):
        """
        [强效特征] 预测同质性：我的预测分布与邻居平均预测分布的相似度
        """
        neighbor_probs = self.aggregate_neighbors(probs, edge_index, num_nodes)
        p_norm = F.normalize(probs, p=2, dim=1)
        n_norm = F.normalize(neighbor_probs, p=2, dim=1)
        homophily = (p_norm * n_norm).sum(dim=1).view(-1, 1)
        return homophily

    def construct_features(self, logits, mlp_logits, robust_scores, edge_index, num_nodes, ablation="none"):
        """
        [Compact & Interaction] 38维特征构造。
        同时生成 feature_names 用于可解释性分析。
        """
        torch.cuda.empty_cache()
        gc.collect()

        # 1. 解包基础稳健特征 (8维)
        s_dropout, s_func, s_struct, s_dist, s_disagree, s_model_bias, s_homophily, s_mlp_entropy = robust_scores

        # Log 平滑
        s_func = torch.log1p(s_func)
        s_struct = torch.log1p(s_struct)
        s_dist = torch.log1p(s_dist)

        # 2. 计算统计不确定性指标
        probs = F.softmax(logits.cpu(), dim=1)

        # (A) Gini Impurity
        gini = 1.0 - torch.sum(probs ** 2, dim=1).view(-1, 1)
        # (B) Margin
        sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
        margin = (sorted_probs[:, 0] - sorted_probs[:, 1]).view(-1, 1)
        # (C) Entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1).view(-1, 1)
        # (D) Max Prob
        max_prob = sorted_probs[:, 0].view(-1, 1)

        # 3. 组合 "自身特征" (Self) - 12维
        scalar_features = torch.stack([
            s_dropout, s_func, s_struct, s_dist,
            s_disagree, s_model_bias, s_homophily, s_mlp_entropy
        ], dim=1)

        self_features = torch.cat([scalar_features, gini, margin, entropy, max_prob], dim=1)

        # 4. 上下文特征 (Context) - 12维
        neighbor_features = self.aggregate_neighbors(self_features, edge_index, num_nodes)

        # 5. 差值特征 (Contrast) - 12维
        diff_features = self_features - neighbor_features

        # 6. 预测同质性 & 度数 - 2维
        pred_homophily = self.get_prediction_homophily(probs, edge_index, num_nodes)
        deg_raw = degree(edge_index.cpu()[1], num_nodes, dtype=torch.float)
        deg_log = torch.log1p(deg_raw).view(-1, 1)

        # 最终拼接 - 38维
        final_features = torch.cat([
            self_features,
            neighbor_features,
            diff_features,
            pred_homophily,
            deg_log
        ], dim=1)

        # --- 生成特征名称 (用于 SHAP 解释) ---
        base_names = [
            "Uncertainty", "Grad_Sens", "Struct_Sens", "OOD_Dist",
            "Neighbor_Disagree", "Model_Bias", "Homophily", "MLP_Entropy",
            "Gini", "Margin", "Entropy", "Max_Prob"
        ]
        context_names = [f"Ctx_{n}" for n in base_names]
        diff_names = [f"Diff_{n}" for n in base_names]
        other_names = ["Pred_Homophily", "Log_Degree"]

        self.feature_names = base_names + context_names + diff_names + other_names

        return final_features.numpy()

    def fit_ensemble(self, train_feats, train_y, sample_weights=None, n_estimators=5):
        """
        Bagging 集成训练 (加权采样)
        """
        self.models = []
        num_samples = len(train_y)
        num_pos = np.sum(train_y)
        num_neg = len(train_y) - num_pos

        scale_weight = min(num_neg / num_pos if num_pos > 0 else 1.0, 3.0)

        if sample_weights is None:
            sample_weights = np.ones(num_samples, dtype=np.float32)

        # 计算采样概率
        sum_w = np.sum(sample_weights)
        sample_probs = sample_weights / sum_w if sum_w > 0 else None

        base_params = {
            'n_estimators': 80,
            'max_depth': 4,
            'learning_rate': 0.05,
            'min_child_weight': 3,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'objective': 'binary:logistic',
            'scale_pos_weight': scale_weight,
            'n_jobs': 1,
            'tree_method': 'hist',
            'random_state': 42
        }

        indices = np.arange(num_samples)

        for i in range(n_estimators):
            # 加权 Bootstrap
            bootstrap_idx = np.random.choice(indices, size=num_samples, replace=True, p=sample_probs)
            X_sample = train_feats[bootstrap_idx]
            y_sample = train_y[bootstrap_idx]

            params = base_params.copy()
            params['random_state'] = 42 + i

            clf = XGBClassifier(**params)
            clf.fit(X_sample, y_sample)
            self.models.append(clf)

    def predict_ensemble(self, test_feats):
        if len(self.models) == 0: return np.zeros(len(test_feats))
        probs_sum = np.zeros(len(test_feats))
        for clf in self.models:
            probs_sum += clf.predict_proba(test_feats)[:, 1]
        return probs_sum / len(self.models)

    def get_feature_names(self):
        return self.feature_names