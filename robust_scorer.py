import torch
import torch.nn.functional as F
import numpy as np
import gc
from torch_geometric.utils import degree
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.nn import MLP
from torch.utils.data import DataLoader


class RobustScorer:
    def __init__(self, args, num_features, num_classes, device):
        self.args = args
        self.device = device
        self.num_classes = num_classes

        # 辅助 MLP 模型 (用于提取 Model-Agnostic 特征)
        self.aux_mlp = MLP(channel_list=[num_features, 256, 64, num_classes],
                           dropout=0.5, norm='batch_norm').to(device)
        self.mlp_trained = False

    def train_aux_mlp(self, x, y, train_mask):
        """训练辅助 MLP"""
        if self.mlp_trained: return
        gc.collect()
        torch.cuda.empty_cache()

        train_idx = torch.nonzero(train_mask).squeeze()
        # 大图降采样
        if len(train_idx) > 50000:
            perm = torch.randperm(len(train_idx))[:50000]
            train_idx = train_idx[perm]

        optimizer = torch.optim.Adam(self.aux_mlp.parameters(), lr=0.01, weight_decay=5e-4)
        self.aux_mlp.train()

        if x.shape[0] > 50000:
            x_device, y_device, batch_size = 'cpu', 'cpu', 1024
        else:
            x_device, y_device, batch_size = self.device, self.device, 4096

        train_idx = train_idx.to(x_device)

        for epoch in range(50):
            perm = torch.randperm(len(train_idx))
            for i in range(0, len(train_idx), batch_size):
                idx = train_idx[perm[i:i + batch_size]]
                batch_x = x[idx].to(self.device)
                batch_y = y[idx].to(self.device)
                optimizer.zero_grad()
                out = self.aux_mlp(batch_x)
                loss = F.cross_entropy(out, batch_y)
                loss.backward()
                optimizer.step()
        self.mlp_trained = True

    def get_mlp_logits(self, x):
        """获取 MLP 的原始 Logits"""
        self.aux_mlp.eval()
        logits_list = []
        if x.size(0) < 50000:
            with torch.no_grad():
                return self.aux_mlp(x.to(self.device)).cpu()

        loader = DataLoader(torch.arange(x.size(0)), batch_size=8192)
        with torch.no_grad():
            for idx in loader:
                batch_x = x[idx].to(self.device)
                logits_list.append(self.aux_mlp(batch_x).cpu())
        return torch.cat(logits_list, dim=0)

    def get_mc_dropout_uncertainty(self, model, data, num_samples=5):
        """MC Dropout 不确定性"""
        gc.collect()
        torch.cuda.empty_cache()
        model.train()  # 必须开启 train 模式以启用 Dropout

        if data.num_nodes < 50000:
            probs_list = []
            x = data.x.to(self.device)
            edge_index = data.edge_index.to(self.device)
            with torch.no_grad():
                for _ in range(num_samples):
                    out = model(x, edge_index)
                    probs_list.append(F.softmax(out, dim=1).cpu())
            # 计算方差总和
            variance = torch.var(torch.stack(probs_list), dim=0).sum(dim=1)
            return variance
        else:
            # 大图分批处理
            loader = DataLoader(torch.arange(data.num_nodes), batch_size=4096)
            sum_probs = torch.zeros((data.num_nodes, self.num_classes), device='cpu')
            for i in range(num_samples):
                probs_list = []
                with torch.no_grad():
                    for idx in loader:
                        batch_x = data.x[idx].to(self.device)
                        empty_edge = torch.empty((2, 0), dtype=torch.long, device=self.device)
                        try:
                            out = model(batch_x, empty_edge)
                        except:
                            out = torch.zeros(len(idx), self.num_classes).to(self.device)
                        probs_list.append(F.softmax(out, dim=1).cpu())
                sum_probs += torch.cat(probs_list, dim=0)
            mean_probs = sum_probs / num_samples
            return -torch.sum(mean_probs * torch.log(mean_probs + 1e-9), dim=1)

    def get_gradient_sensitivity(self, model, data, preds):
        """
        [选题4 核心 - 白盒特征]
        计算输入特征的梯度敏感度。
        这是一个强力的白盒指标，用于探测对抗性脆弱点。
        如果一个节点只需要微小的特征扰动就能改变 Loss，说明它是脆弱的。
        """
        model.eval()
        # 仅对小图计算，大图计算梯度开销过大
        if data.num_nodes < 50000:
            x = data.x.to(self.device).clone().detach().requires_grad_(True)
            edge_index = data.edge_index.to(self.device)
            pseudo_labels = preds.to(self.device)
            try:
                out = model(x, edge_index)
                loss = F.cross_entropy(out, pseudo_labels)
                model.zero_grad()
                if x.grad is not None: x.grad.zero_()
                loss.backward()
                if x.grad is not None:
                    # L2 Norm of Gradient
                    return torch.norm(x.grad, p=2, dim=1).cpu()
            except:
                pass
            return torch.zeros(x.size(0))
        return torch.zeros(data.num_nodes)

    def get_structure_sensitivity(self, model, data, preds):
        """结构梯度敏感度 (检测对边的依赖程度)"""
        if data.num_nodes > 50000: return torch.zeros(data.num_nodes)

        model_type = self.args.model_type
        if model_type not in ["GCN", "TAGCN", "SAGE", "GraphSAGE"]:
            return torch.zeros(data.num_nodes)

        model.eval()
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        pseudo_labels = preds.to(self.device)

        edge_weight = torch.ones(edge_index.shape[1], device=self.device, requires_grad=True)

        try:
            out = model(x, edge_index, edge_weight=edge_weight)
            loss = F.cross_entropy(out, pseudo_labels)
            model.zero_grad()
            loss.backward()
            if edge_weight.grad is not None:
                return scatter_add(edge_weight.grad.abs(), edge_index[1], dim=0, dim_size=data.num_nodes).cpu()
        except:
            return torch.zeros(data.num_nodes)

        return torch.zeros(data.num_nodes)

    def get_neighbor_disagreement(self, preds, edge_index, num_nodes):
        """邻居不一致比例"""
        row, col = edge_index.cpu()
        src_pred = preds.cpu()[row]
        dst_pred = preds.cpu()[col]
        is_diff = (src_pred != dst_pred).float()
        diff_count = scatter_add(is_diff, col, dim=0, dim_size=num_nodes)
        deg = degree(col, num_nodes, dtype=torch.float)
        return diff_count / (deg + 1e-9)

    def _calc_dist_score_light(self, model, data, preds):
        """OOD 距离"""
        model.eval()
        with torch.no_grad():
            if data.num_nodes > 50000: return torch.zeros(data.num_nodes)
            try:
                emb = model.get_embedding(data.x.to(self.device), data.edge_index.to(self.device)).cpu().numpy()
            except:
                emb = data.x.cpu().numpy()

        train_mask = data.train_mask.cpu().numpy()
        labels = data.y.cpu().numpy()
        preds_np = preds.cpu().numpy()

        class_means = {}
        norm_emb = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
        normalized_emb = emb / norm_emb

        for c in range(self.num_classes):
            c_mask = (labels == c) & train_mask
            if c_mask.sum() > 0:
                mean_vec = np.mean(normalized_emb[c_mask], axis=0)
                class_means[c] = mean_vec / (np.linalg.norm(mean_vec) + 1e-9)
            else:
                class_means[c] = np.zeros(normalized_emb.shape[1])

        scores = np.zeros(data.num_nodes)
        for c in range(self.num_classes):
            mask = (preds_np == c)
            if mask.sum() > 0:
                sim = np.dot(normalized_emb[mask], class_means[c])
                scores[mask] = 1 - sim

        return torch.tensor(scores, dtype=torch.float)

    def get_model_disagreement_soft(self, preds, mlp_logits):
        """GNN 与 MLP 的预测分歧"""
        if mlp_logits is None:
            return torch.zeros(preds.size(0))

        mlp_probs = F.softmax(mlp_logits.cpu(), dim=1)
        gnn_pred_indices = preds.cpu()
        prob_at_gnn_pred = mlp_probs.gather(1, gnn_pred_indices.view(-1, 1)).squeeze()
        return 1.0 - prob_at_gnn_pred

    def get_feature_homophily(self, data):
        """特征同质性"""
        x = data.x.cpu()
        edge_index = data.edge_index.cpu()
        src_idx, target_idx = edge_index[0], edge_index[1]

        mean_neighbor_x = scatter_mean(x[src_idx], target_idx, dim=0, dim_size=data.num_nodes)

        x_norm = F.normalize(x, p=2, dim=1)
        mean_norm = F.normalize(mean_neighbor_x, p=2, dim=1)

        homophily = (x_norm * mean_norm).sum(dim=1)
        homophily[homophily == 0] = 0.5
        return homophily

    def get_scores(self, model, data, preds, mlp_logits=None):
        """
        主入口：计算 8 个核心稳健特征。
        包括白盒特征 (Gradient Sensitivity) 和黑盒/灰盒特征。
        """
        s_dropout = self.get_mc_dropout_uncertainty(model, data).cpu()
        s_func = self.get_gradient_sensitivity(model, data, preds).cpu()
        s_struct = self.get_structure_sensitivity(model, data, preds).cpu()
        s_dist = self._calc_dist_score_light(model, data, preds).cpu()
        s_disagree = self.get_neighbor_disagreement(preds, data.edge_index, data.num_nodes).cpu()
        s_model_bias = self.get_model_disagreement_soft(preds, mlp_logits).cpu()
        s_homophily = self.get_feature_homophily(data).cpu()

        # 8. Model-Agnostic MLP Entropy
        if mlp_logits is not None:
            mlp_probs = F.softmax(mlp_logits, dim=1)
            s_mlp_entropy = -torch.sum(mlp_probs * torch.log(mlp_probs + 1e-9), dim=1).cpu()
        else:
            s_mlp_entropy = torch.zeros(preds.size(0))

        return s_dropout, s_func, s_struct, s_dist, s_disagree, s_model_bias, s_homophily, s_mlp_entropy