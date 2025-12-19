import torch
import os
from torch_geometric.datasets import Planetoid, Reddit, Flickr
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.transforms import NormalizeFeatures


# 注意：移除了 ToSparseTensor 的导入，因为它导致了 ogbn-products 的兼容性问题

def load_data(args):
    print(f"   [Data] Loading {args.dataset} from {args.root}...")

    if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root=args.root, name=args.dataset, transform=NormalizeFeatures())
        data = dataset[0]

    elif args.dataset == 'Reddit':
        path = os.path.join(args.root, 'Reddit')
        dataset = Reddit(root=path)
        data = dataset[0]

    elif args.dataset == 'Flickr':
        path = os.path.join(args.root, 'Flickr')
        dataset = Flickr(root=path)
        data = dataset[0]

    elif args.dataset == 'ogbn-products':
        # 【核心修复】移除 transform=ToSparseTensor()
        # 这样数据会保留 edge_index 格式，适配 main.py 中的 NeighborLoader 和模型调用
        dataset = PygNodePropPredDataset(name='ogbn-products', root=args.root)
        data = dataset[0]

        # OGB 的标签通常是 (N, 1)，转为 (N,)
        if data.y.dim() > 1:
            data.y = data.y.squeeze()

        split_idx = dataset.get_idx_split()

        # 手动构建 mask (ogb 默认不提供 mask)
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

        train_mask[split_idx['train']] = True
        val_mask[split_idx['valid']] = True
        test_mask[split_idx['test']] = True

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

    else:
        raise NotImplementedError(f"Dataset {args.dataset} not supported.")

    # 统一补充属性
    if not hasattr(dataset, 'num_classes'):
        # 部分数据集可能没有显式 num_classes
        dataset.num_classes = int(data.y.max().item()) + 1

    if not hasattr(dataset, 'num_features'):
        dataset.num_features = data.x.shape[1]

    # 确保 mask 存在
    if not hasattr(data, 'train_mask'):
        raise ValueError("Data object missing train_mask!")

    return dataset, data