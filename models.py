import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TAGConv, SAGEConv
from torch_geometric.nn import Linear


class BaseGNN(nn.Module):
    def __init__(self):
        super().__init__()

    def get_embedding(self, x, edge_index):
        """
        用于 RobustScorer 计算 OOD 距离和特征分布。
        必须返回隐层的高质量表示。
        """
        raise NotImplementedError


# --- 经典模型优化 (GCN, GAT, TAGCN) ---

class GCN(BaseGNN):
    def __init__(self, in_feats, hidden_dim, out_feats, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_dim)
        # [优化] 加入 BatchNorm，防止过度平滑，保持特征锐度
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_feats)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        # [优化] Input Dropout: 模拟特征噪声，提升鲁棒性，对齐 GAT 的机制
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)  # BN
        x = F.relu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index, edge_weight)

    def get_embedding(self, x, edge_index):
        # 确保 Embedding 与 forward 逻辑一致 (包含 BN 和 ReLU)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        return F.relu(x)


class GAT(BaseGNN):
    """
    GAT 本身结构已经很优秀 (包含 Input Dropout)，保持不变。
    """

    def __init__(self, in_feats, hidden_dim, out_feats, dropout, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_feats, hidden_dim, heads=heads, dropout=dropout)
        # 注意: output dimension is hidden_dim * heads
        self.conv2 = GATConv(hidden_dim * heads, out_feats, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        # Input Dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)

    def get_embedding(self, x, edge_index):
        return F.elu(self.conv1(x, edge_index))


class TAGCN(BaseGNN):
    def __init__(self, in_feats, hidden_dim, out_feats, dropout, K=2):
        super().__init__()
        self.conv1 = TAGConv(in_feats, hidden_dim, K=K)
        # [优化] 加入 BatchNorm
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = TAGConv(hidden_dim, out_feats, K=K)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        # [优化] Input Dropout
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index, edge_weight)

    def get_embedding(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        return F.relu(x)


# --- 大图模型优化 ---

class GraphSAGE(BaseGNN):
    def __init__(self, in_feats, hidden_dim, out_feats, dropout):
        super().__init__()
        self.conv1 = SAGEConv(in_feats, hidden_dim)
        # [优化] SAGE 也加入 BN
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_feats)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        # [优化] Input Dropout
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)

    def get_embedding(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        return F.relu(x)


class SIGN(BaseGNN):
    def __init__(self, in_feats, hidden_dim, out_feats, dropout, K=3):
        super().__init__()
        self.K = K
        # 1. 传播层 (模拟预计算)
        self.prop_layers = nn.ModuleList()
        for _ in range(K):
            self.prop_layers.append(
                SAGEConv(in_channels=in_feats, out_channels=in_feats,
                         root_weight=False, bias=False)
            )

        # 2. MLP 增强版 (Linear -> LayerNorm -> ReLU -> Dropout -> Linear)
        # 输入维度是 (K + 1) * in_feats
        input_dim = (K + 1) * in_feats

        self.lin1 = Linear(input_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)  # SIGN 原版已有 LayerNorm，保持
        self.lin2 = Linear(hidden_dim, out_feats)

        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        # 特征传播
        zs = [x]
        curr = x
        for i in range(self.K):
            curr = self.prop_layers[i](curr, edge_index)
            zs.append(curr)

        # 拼接
        x = torch.cat(zs, dim=1)

        # MLP 变换
        x = self.lin1(x)
        x = self.ln(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return x

    def get_embedding(self, x, edge_index):
        zs = [x]
        curr = x
        for i in range(self.K):
            curr = self.prop_layers[i](curr, edge_index)
            zs.append(curr)
        x = torch.cat(zs, dim=1)

        x = self.lin1(x)
        x = self.ln(x)
        return F.relu(x)


class EnGCN(BaseGNN):
    def __init__(self, in_feats, hidden_dim, out_feats, dropout):
        super().__init__()
        self.mlp1 = nn.Sequential(Linear(in_feats, hidden_dim), nn.ReLU(), Linear(hidden_dim, out_feats))
        self.mlp2 = nn.Sequential(Linear(in_feats, hidden_dim), nn.ReLU(), Linear(hidden_dim, out_feats))
        self.conv_1hop = SAGEConv(in_feats, in_feats, root_weight=False, bias=False)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        # Input Dropout 对 Boosting 类方法也有效
        x = F.dropout(x, p=self.dropout, training=self.training)
        out1 = self.mlp1(x)
        x_1hop = self.conv_1hop(x, edge_index)
        out2 = self.mlp2(x_1hop)
        return out1 + out2

    def get_embedding(self, x, edge_index):
        x_1hop = self.conv_1hop(x, edge_index)
        # 取 MLP2 的中间层作为 Embedding
        return F.relu(self.mlp2[0](x_1hop))


def get_model(args, num_features, num_classes):
    if args.model_type == "SAGE" or args.model_type == "ClusterGCN":
        return GraphSAGE(num_features, args.hidden_dim, num_classes, args.dropout)
    elif args.model_type == "SIGN":
        return SIGN(num_features, args.hidden_dim, num_classes, args.dropout, K=args.sign_hops)
    elif args.model_type == "EnGCN":
        return EnGCN(num_features, args.hidden_dim, num_classes, args.dropout)
    elif args.model_type == "GCN":
        return GCN(num_features, args.hidden_dim, num_classes, args.dropout)
    elif args.model_type == "GAT":
        return GAT(num_features, args.hidden_dim, num_classes, args.dropout, heads=args.gat_heads)
    elif args.model_type == "TAGCN":
        return TAGCN(num_features, args.hidden_dim, num_classes, args.dropout, K=args.tagcn_k)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")