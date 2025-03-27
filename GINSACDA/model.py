import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SortPooling, SumPooling, AvgPooling, MaxPooling
from dgl.nn.pytorch import GATConv, GraphConv, ChebConv, AGNNConv, APPNPConv, RelGraphConv, DotGatConv
from GCN_layer import GCN2Conv
from GIN_layer import GINConv
from GraphSAGE_layer import SAGEConv
from GAT_layer_v2 import GATv2Conv
from torch.nn import Dropout, ELU, Sequential, Linear, ReLU
import dgl
from utils import parse_arguments

args = parse_arguments()


class Model(nn.Module):
    def __init__(self, args, G, model_name, num_diseases, num_rna, num_layers, in_dim, hidden_units, gnn_type,
                        use_attribute,device, dropout=0.5, max_z=4000, alpha=None, hop_num=None):

        super(Model, self).__init__()
        self.G = G
        self.num_diseases = num_diseases
        self.num_rna = num_rna
        self.num_layers = num_layers
        self.model_name = model_name
        self.dropout = dropout
        self.gnn_type = gnn_type
        self.device = device
        self.in_dim = in_dim
        self.hidden_units = hidden_units
        self.use_attribute = use_attribute
        self.c_fc = nn.Linear(G.ndata['c_feat'].shape[1], in_dim, bias=False)
        self.d_fc = nn.Linear(G.ndata['d_feat'].shape[1], in_dim, bias=False)

        self.z_embedding = nn.Embedding(max_z, in_dim)

        if model_name == 'gin':
            self.model = GNNModel(
                num_layers=args.num_layers,
                hidden_units=args.hidden_units,
                gnn_type=args.gnn_type,
                alpha=alpha,
                hop_num=hop_num,
                node_attributes=None,
                edge_weights=None,
                node_embedding=None,
                use_embedding=False,
                num_nodes=G.num_nodes(),
                dropout=args.dropout,
                device=device
            )

        else:
            raise ValueError('Model error')

    def forward(self, g, z, node_id=None, edge_id=None):
        z_emb = self.z_embedding(z.long())
        if self.use_attribute:
            disease_mask = node_id <= self.num_diseases
            rna_mask = node_id > self.num_diseases
            d_feat = self.d_fc(self.G.ndata['d_feat'][node_id][disease_mask])
            c_feat = self.c_fc(self.G.ndata['c_feat'][node_id][rna_mask])
            feat = torch.cat([d_feat, c_feat])
            x = torch.cat([z_emb, feat], 1)
        else:
            x = z_emb

        if self.model_name == 'vgae':
            logits = self.model(g, x, self.device)
        else:
            logits = self.model(g, x)
        return logits



class GNNModel(torch.nn.Module):
    def __init__(self, num_layers, node_attributes, use_embedding, edge_weights, num_nodes, hidden_units, device,
                 dropout=0., gnn_type='gin', residual=True, use_mlp=False, join_with_mlp=False,
                 node_embedding=None, max_z=1000, alpha=None, hop_num=None, heads=1):
        super(GNNModel, self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_units = hidden_units
        self.use_attribute = False if node_attributes is None else True
        self.use_embedding = use_embedding
        self.use_edge_weight = False if edge_weights is None else True
        self.alpha = alpha
        self.hop_num = hop_num
        self.heads = heads

        self.z_embedding = nn.Embedding(max_z, hidden_units)
        if node_attributes is not None:
            self.node_attributes_lookup = nn.Embedding.from_pretrained(node_attributes)
            self.node_attributes_lookup.weight.requires_grad = False
        if edge_weights is not None:
            self.edge_weights_lookup = nn.Embedding.from_pretrained(edge_weights)
            self.edge_weights_lookup.weight.requires_grad = False
        if node_embedding is not None:
            self.node_embedding = nn.Embedding.from_pretrained(node_embedding)
            self.node_embedding.weight.requires_grad = False
        elif use_embedding:
            self.node_embedding = nn.Embedding(num_nodes, hidden_units)

        initial_dim = hidden_units
        if self.use_attribute:
            initial_dim += self.node_attributes_lookup.embedding_dim
        if self.use_embedding:
            initial_dim += self.node_embedding.embedding_dim

        self.gnn_type = gnn_type
        self.use_mlp = use_mlp
        self.join_with_mlp = join_with_mlp
        self.normalize_input_columns = True

        self.pooling = MaxPooling()
        self.linear_1 = nn.Linear(hidden_units, hidden_units)
        self.linear_2 = nn.Linear(hidden_units, 1)

        if gnn_type == 'ginsage':
            # self.gin = GINConv(
            #     nn.Sequential(nn.Linear(hidden_units, hidden_units), nn.ReLU(), nn.Linear(hidden_units, hidden_units)))
            self.gin = GINConv(
                nn.Sequential(nn.Linear(hidden_units, hidden_units), nn.ReLU(), nn.Linear(hidden_units, hidden_units)))
            self.sage = SAGEConv(hidden_units, hidden_units, 'mean')
            # self.sage = SAGEConv(hidden_units, hidden_units, 'mean')
            # self.gat = GATv2Conv(hidden_units, hidden_units, self.heads)
            # self.gcn = GCN2Conv(hidden_units, hidden_units)


        elif gnn_type == 'sagegin':
            # self.gcn = GCN2Conv(hidden_units, hidden_units)
            # self.gat = GATv2Conv(hidden_units, hidden_units, self.heads)
            # self.sage = SAGEConv(hidden_units, hidden_units, 'mean')
            self.sage = SAGEConv(hidden_units, hidden_units, 'mean')
            self.gin = GINConv(
                nn.Sequential(nn.Linear(hidden_units, hidden_units), nn.ReLU(), nn.Linear(hidden_units, hidden_units)))
            # self.gin = GINConv(
            #     nn.Sequential(nn.Linear(hidden_units, hidden_units), nn.ReLU(), nn.Linear(hidden_units, hidden_units)))

    def forward(self, graph, x):

        h = x
        if self.use_mlp:
            if self.join_with_mlp:
                h = torch.cat((h, self.mlp(x)), 1)
            else:
                h = self.mlp(x)
        if self.gnn_type == 'ginsage':

            # h = self.gin(graph, h)
            h = self.gin(graph, h)
            h = self.sage(graph, h)
            # h = self.sage(graph, h)
            # h = self.gat(graph, h)
            # h = self.gcn(graph, h, h)

            h = self.pooling(graph, h)
            h = F.relu(self.linear_1(h))
            h = F.dropout(h, p=self.dropout, training=self.training)
            logits = self.linear_2(h)

        elif self.gnn_type == 'sagegin':

            # h = self.gcn(graph, h)
            # h = self.gat(graph, h)
            # h = self.sage(graph, h)
            h = self.sage(graph, h)
            h = self.gin(graph, h)
            # h = self.gin(graph, h)

            h = self.pooling(graph, h)
            h = F.relu(self.linear_1(h))
            h = F.dropout(h, p=self.dropout, training=self.training)
            logits = self.linear_2(h)

        return logits


