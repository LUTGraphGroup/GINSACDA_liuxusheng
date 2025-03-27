import argparse
from scipy.sparse.csgraph import shortest_path
from sklearn import metrics
import numpy as np
import pandas as pd
import torch
import dgl
import math
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator
import scipy.io as scio


def parse_arguments():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(description='SEAL')
    parser.add_argument('--dataset', type=str, default='CircR2Disease')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--model', type=str, default='gin')
    parser.add_argument('--gnn_type', type=str, default='ginsage')
    parser.add_argument('--use_attribute', default=True)
    parser.add_argument('--hop', type=int, default=7)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hop_num', type=int, default=5)
    parser.add_argument('--in_dim', type=int, default=32)
    parser.add_argument('--hidden_units', type=int, default=64)
    parser.add_argument('--sort_k', type=int, default=15)
    parser.add_argument('--pooling', type=str, default='max')
    parser.add_argument('--dropout', type=str, default=0.5)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--neg_samples', type=int, default=1)
    parser.add_argument('--subsample_ratio', type=float, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--random_seed', type=int, default=2024)
    parser.add_argument('--load_dir', type=str, default='./data1.0/CircR2Disease/')
    parser.add_argument('--save_dir', type=str, default='./processed')
    args = parser.parse_args()

    return args


def build_graph(device, directory):
    IC_DO = pd.read_excel(directory + 'DO/integrated_circ_sim.xlsx', header=None)
    IC_DO = pd.read_excel(directory + 'DO/integrated_dise_sim.xlsx', header=None)
    IC_sequence = pd.read_excel(directory + 'CircRNA sequence similarity/LevenshteinSimilar.xlsx',header=None)
    IC_G = pd.read_excel(directory + 'Gaussian interaction profile kernel/circ_gipk.xlsx', header=None)
    ID_G = pd.read_excel(directory + 'Gaussian interaction profile kernel/dis_gipk.xlsx', header=None)
    IC_mesh = pd.read_excel(directory + 'MeSH/integrated_circ_sim.xlsx', header=None)
    ID_mesh = pd.read_excel(directory + 'MeSH/integrated_dise_sim.xlsx', header=None)
    IC = np.hstack((IC_mesh, IC_G))
    ID = np.hstack((ID_mesh, ID_G))
    associations = pd.read_excel(directory + 'Association Matrixs.xlsx', sheet_name='Association Matrix', header=None)

    # IC_G = pd.read_excel('./data1.0/HMDD v3.2/miGIPSim.xlsx', header=None)
    # ID_G = pd.read_excel('./data1.0/HMDD v3.2/disGIPSim.xlsx', header=None)
    # IC_mesh = pd.read_excel('./data1.0/HMDD v3.2/miFunSim.xlsx', header=None)
    # ID_mesh = pd.read_excel('./data1.0/HMDD v3.2/disSeSim.xlsx', header=None)
    # IC = np.hstack((IC_mesh, IC_G))
    # ID = np.hstack((ID_mesh, ID_G))
    # associations = pd.read_excel('./data1.0/HMDD v3.2/hmdd3.2_MDA.xlsx', header=None)

    print("IC.shape", IC.shape)
    print("ID.shape", ID.shape)
    print("associations.shape", associations.shape)

    known_asss = associations.where(associations > 0).stack()  # 筛选数据
    known_associations = pd.concat(
        [pd.DataFrame([[known_asss.index[i][0], known_asss.index[i][1] + IC.shape[0]]], columns=['circrna', 'diseases'])
         for i in range(known_asss.__len__())])

    # 创建异构图g，0为circRNA，1为disease
    g = dgl.DGLGraph().to(device)
    g.add_nodes(IC.shape[0] + ID.shape[0])
    node_type = torch.zeros(g.number_of_nodes(), dtype=torch.int64).to(device)
    node_type[: ID.shape[0]] = 1
    g.ndata['type'] = node_type

    # 添加疾病特征
    d_feat = torch.zeros(g.number_of_nodes(), ID.shape[1]).to(device)
    d_feat[: ID.shape[0], :] = torch.from_numpy(ID.astype('float32'))
    # d_feat[:ID.shape[0], :] = torch.from_numpy(ID.to_numpy().astype('float32'))
    g.ndata['d_feat'] = d_feat

    # 添加circRNA特征
    c_feat = torch.zeros(g.number_of_nodes(), IC.shape[1]).to(device)
    c_feat[ID.shape[0]: ID.shape[0] + IC.shape[0], :] = torch.from_numpy(IC.astype('float32')).to(device)
    # c_feat[ID.shape[0]: ID.shape[0] + IC.shape[0], :] = torch.from_numpy(IC.to_numpy().astype('float32')).to(device)
    g.ndata['c_feat'] = c_feat

    # 创建节点ID列表
    disease_ids = list(range(1, ID.shape[0] + 1))
    circrna_ids = list(range(1, IC.shape[0] + 1))

    # 添加边
    g.add_edges(torch.tensor(known_associations['circrna'].values).to(device),
                torch.tensor(known_associations['diseases'].values).to(device))

    g = dgl.add_reverse_edges(g)
    return g, IC, ID


def get_split_edge(g, train_idx, test_idx):
    src, dst = g.edges()
    mask = src < dst
    src, dst = src[mask], dst[mask]

    # 检查索引有效性
    if train_idx.max() >= src.size(0) or train_idx.min() < 0:
        raise ValueError("train_idx contains out-of-bounds indices.")

    # 将 test_idx 划分为验证集和测试集
    s, d = src[test_idx[:math.floor(len(test_idx)/2)]], dst[test_idx[:math.floor(len(test_idx)/2)]]
    val_pos_edge_index = torch.cat([torch.stack([s, d], dim=0), torch.stack([d, s], dim=0)], dim=1)
    s, d = src[test_idx[math.floor(len(test_idx)/2):]], dst[test_idx[math.floor(len(test_idx)/2):]]
    test_pos_edge_index = torch.cat([torch.stack([s, d], dim=0), torch.stack([d, s], dim=0)], dim=1)
    # 训练集
    s, d = src[train_idx], dst[train_idx]
    train_pos_edge_index = torch.cat([torch.stack([s, d], dim=0), torch.stack([d, s], dim=0)], dim=1)
    # 生成负样本
    n_src, n_dst = dgl.sampling.global_uniform_negative_sampling(g, int(g.num_edges()/2))
    val_neg_edge_index = torch.cat([torch.stack([n_src[test_idx[:math.floor(len(test_idx)/2)]], n_dst[test_idx[:math.floor(len(test_idx)/2)]]]), torch.stack([n_dst[test_idx[:math.floor(len(test_idx)/2)]], n_src[test_idx[:math.floor(len(test_idx)/2)]]], dim=0)], dim=1)
    test_neg_edge_index = torch.cat([torch.stack([n_src[test_idx[math.floor(len(test_idx)/2)]:], n_dst[test_idx[math.floor(len(test_idx)/2)]:]]), torch.stack([ n_dst[test_idx[math.floor(len(test_idx)/2)]:], n_src[test_idx[math.floor(len(test_idx)/2)]:]], dim=0)], dim=1)
    train_neg_edge_index = torch.cat([torch.stack([n_src[train_idx], n_dst[train_idx]]), torch.stack([n_dst[train_idx], n_src[train_idx]], dim=0)], dim=1)

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = train_pos_edge_index.t()
    split_edge['train']['edge_neg'] = train_neg_edge_index.t()
    split_edge['valid']['edge'] = val_pos_edge_index.t()
    split_edge['valid']['edge_neg'] = val_neg_edge_index.t()
    split_edge['test']['edge'] = test_pos_edge_index.t()
    split_edge['test']['edge_neg'] = test_neg_edge_index.t()

    return split_edge


def load_ogb_dataset(dataset):

    dataset = DglLinkPropPredDataset(name=dataset)
    split_edge = dataset.get_edge_split()
    graph = dataset[0]

    return graph, split_edge


def drnl_node_labeling(subgraph, src, dst):

    adj = subgraph.adj().to_dense().cpu().numpy()
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst - 1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    z[torch.isnan(z)] = 0.

    return z.to(torch.long)


def path_length_labeling(subgraph, src):
    adj = subgraph.adj().to_dense().cpu().numpy()

    # 计算从源节点到所有其他节点的最短路径
    dist = shortest_path(adj, directed=False, unweighted=True, indices=src)

    # 转换为张量
    z = torch.from_numpy(dist)

    return z.to(torch.long)


def centrality_labeling(subgraph):
    # 获取邻接矩阵
    adj = subgraph.adj().to_dense().cpu().numpy()

    # 计算每个节点的度数
    degrees = np.array(adj.sum(axis=0)).flatten()

    # 将度数转换为张量
    z = torch.from_numpy(degrees)

    return z.to(torch.float)


def best_threshold(fpr, tpr, thresholds):
    ths = thresholds
    diffs = list(tpr - fpr)
    max_diff = max(diffs)
    optimal_idx = diffs.index(max_diff)
    optimal_th = ths[optimal_idx]
    return optimal_th


def evaluate_roc(y_pred_pos, y_pred_neg):
    y_pred_pos_numpy = y_pred_pos.cpu().detach().numpy()
    y_pred_neg_numpy = y_pred_neg.cpu().detach().numpy()

    # 创建真实标签
    y_true = np.concatenate([np.ones(len(y_pred_pos_numpy)), np.zeros(len(y_pred_neg_numpy))]).astype(np.int32)
    y_pred = np.concatenate([y_pred_pos_numpy, y_pred_neg_numpy])
    # y_pred = Abnormal_data_handling(y_pred)

    # 计算FPR, TPR和阈值
    fpr_, tpr_, thresholds = metrics.roc_curve(y_true, y_pred)
    best_th = best_threshold(fpr_, tpr_, thresholds)

    # 计算AUC和其他指标
    auc = metrics.roc_auc_score(y_true, y_pred)
    aupr = metrics.average_precision_score(y_true, y_pred)

    # 生成预测标签
    pred_label = [0 if j < best_th else 1 for j in y_pred]

    # 计算各种指标
    acc = metrics.accuracy_score(y_true, pred_label)
    pre = metrics.precision_score(y_true, pred_label)
    recall = metrics.recall_score(y_true, pred_label)
    f1 = metrics.f1_score(y_true, pred_label)

    # 计算精确率-召回率曲线
    precision_, recall_, _ = metrics.precision_recall_curve(y_true, y_pred)
    prc = metrics.auc(recall_, precision_)

    return auc, aupr, acc, pre, recall, precision_, recall_, f1, prc, fpr_, tpr_,

