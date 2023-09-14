import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision.models as models
from torch.nn import Linear
# from src.functions.feat_pred_fuc.layers import AttentionConv


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, init='xavier'):
        super(GraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.FloatTensor(in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_channels))
        else:
            self.register_parameter('bias', None)
        if init == 'uniform':
            self.reset_parameters_uniform()
        elif init == 'xavier':
            self.reset_parameters_xavier()
        else:
            raise NotImplementedError

    def reset_parameters_uniform(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02)  # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def forward(self, x, adj):
        x = torch.matmul(x, self.weight)
        out = torch.spmm(adj, x)
        if self.bias is not None:
            return out + self.bias
        else:
            return out


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout=0.5, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        # self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU()

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class GraphAttention(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.1, concat=True, init="xavier"):
        super(GraphAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # self.A = nn.Parameter(torch.zeros(size=(in_features, in_features)))

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()
        if init == 'uniform':
            #print("| Uniform Initialization")
            self.reset_parameters_uniform()
        elif init == 'xavier':
            #print("| Xavier Initialization")
            self.reset_parameters_xavier()
        elif init == 'kaiming':
            #print("| Kaiming Initialization")
            self.reset_parameters_kaiming()
        else:
            raise NotImplementedError

    def reset_parameters_uniform(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        if self.a is not None:
            self.a.data.uniform_(-stdv, stdv)

    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.W.data, gain=0.02) # Implement Xavier Uniform
        if self.a is not None:
            nn.init.constant_(self.a.data, 0.0)

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.W.data, a=0, mode='fan_in')
        if self.a is not None:
            nn.init.constant_(self.a.data, 0.0)

    def forward(self, input, adj, score=None):
        # dv = 'cuda' if input.is_cuda else 'cpu'
        dv = input.device

        N = input.size()[0]
        edge = torch.nonzero(adj).t()
        # if score!=None:
        #     input = input + self.leakyrelu(torch.mm(input * score[:, None], self.A))
        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        if score!=None:
            score_mtx = (score[:,None] * score[None])**0.5
            edge_e = edge_e * score_mtx[edge[0,:], edge[1,:]]
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum + 0.00001)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime, edge_e

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class TopoGCN(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGCN, self).__init__()
        self.args = args
        self.info_dim = 1 + 3 + args.goal_type_num # visited, position, goal text feat dim
        if args.use_cm_score:
            self.info_dim += 1 # cm_score
        # self.feat_dim = args.vis_feat_dim + 1 + 3 + args.goal_type_num
        self.feat_dim = args.vis_feat_dim + self.info_dim


        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

        self.feat_enc = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            # nn.Sigmoid()
        )

        self.graph_conv1 = GraphConv(hidden_size, hidden_size)
        self.graph_conv2 = GraphConv(hidden_size, hidden_size)
        self.graph_conv3 = GraphConv(hidden_size, hidden_size)
        self.graph_conv4 = GraphConv(hidden_size, hidden_size)
        self.graph_conv5 = GraphConv(hidden_size, hidden_size)
        # self.distance_layer = GraphConv(hidden_size, 1)

        # self.distance_layers1 = GraphConv(hidden_size, hidden_size)
        # self.distance_layers2 = GraphConv(hidden_size, hidden_size)
        # self.distance_layers3 = GraphConv(hidden_size, 1)


    def forward(self, feat, adj):

        # feat_x = self.feat_enc(feat)

        feat_x = self.dropout(self.relu(self.graph_conv1(feat, adj)))
        feat_x = self.dropout(self.relu(self.graph_conv2(feat_x, adj)))
        feat_x = self.dropout(self.relu(self.graph_conv3(feat_x, adj)))
        feat_x = self.dropout(self.relu(self.graph_conv4(feat_x, adj)))
        feat_x = self.dropout(self.relu(self.graph_conv5(feat_x, adj)))

        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist


class TopoGCN_v2(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGCN_v2, self).__init__()
        self.args = args
        self.info_dim = 1 + 3 + args.vis_feat_dim  # visited, position, goal text feat dim
        if args.use_cm_score:
            self.info_dim += 1  # cm_score
        self.feat_dim = args.vis_feat_dim + self.info_dim

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

        self.feat_enc = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            # nn.Sigmoid()
        )

        self.graph_conv1 = GraphConv(hidden_size + self.info_dim, hidden_size)
        self.graph_conv2 = GraphConv(hidden_size + self.info_dim, hidden_size)
        self.graph_conv3 = GraphConv(hidden_size + self.info_dim, hidden_size)
        self.graph_conv4 = GraphConv(hidden_size + self.info_dim, hidden_size)
        self.graph_conv5 = GraphConv(hidden_size + self.info_dim, hidden_size)
        # self.distance_layer = GraphConv(hidden_size, 1)

        # self.distance_layers1 = GraphConv(hidden_size, hidden_size)
        # self.distance_layers2 = GraphConv(hidden_size, hidden_size)
        # self.distance_layers3 = GraphConv(hidden_size, 1)

    def forward(self, feat, goal_feat, info_feat, adj):
        feat_x = torch.cat([feat, goal_feat, info_feat], dim=-1)
        feat_x = self.feat_enc(feat_x)

        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        feat_x = self.dropout(self.relu(self.graph_conv1(feat_x, adj)))
        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        feat_x = self.dropout(self.relu(self.graph_conv2(feat_x, adj)))
        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        feat_x = self.dropout(self.relu(self.graph_conv3(feat_x, adj)))
        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        feat_x = self.dropout(self.relu(self.graph_conv4(feat_x, adj)))
        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        feat_x = self.dropout(self.relu(self.graph_conv5(feat_x, adj)))

        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist



class TopoGCN_v3(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGCN_v3, self).__init__()
        self.args = args
        self.info_dim = 1 + 3 + args.vis_feat_dim  # visited, position, goal text feat dim
        if args.use_cm_score:
            self.info_dim += 1  # cm_score
        self.feat_dim = args.vis_feat_dim + self.info_dim

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

        self.feat_enc = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.value_layer = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            # nn.Sigmoid()
        )

        self.graph_attention1 = SpGraphAttentionLayer(hidden_size, hidden_size)
        self.graph_attention2 = SpGraphAttentionLayer(hidden_size, hidden_size)
        self.graph_attention3 = SpGraphAttentionLayer(hidden_size, hidden_size)
        self.graph_attention4 = SpGraphAttentionLayer(hidden_size, hidden_size)
        self.graph_attention5 = SpGraphAttentionLayer(hidden_size, hidden_size)


    def forward(self, feat, goal_feat, info_feat, adj):
        feat_x = torch.cat([feat, goal_feat, info_feat], dim=-1)
        feat_x = self.feat_enc(feat_x)

        feat_x = self.dropout(self.relu(self.graph_attention1(feat_x, adj)))
        feat_x = self.dropout(self.relu(self.graph_attention2(feat_x, adj)))
        feat_x = self.dropout(self.relu(self.graph_attention3(feat_x, adj)))
        feat_x = self.dropout(self.relu(self.graph_attention4(feat_x, adj)))
        feat_x = self.dropout(self.relu(self.graph_attention5(feat_x, adj)))
        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist

class TopoGCN_v3_skip(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGCN_v3_skip, self).__init__()
        self.args = args
        self.info_dim = 1 + 3 + args.vis_feat_dim  # visited, position, goal text feat dim
        if args.use_cm_score:
            self.info_dim += 1  # cm_score
        self.feat_dim = args.vis_feat_dim + self.info_dim

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

        self.feat_enc = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.value_layer = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            # nn.Sigmoid()
        )

        self.graph_attention1 = SpGraphAttentionLayer(hidden_size + self.info_dim, hidden_size)
        self.graph_attention2 = SpGraphAttentionLayer(hidden_size + self.info_dim, hidden_size)
        self.graph_attention3 = SpGraphAttentionLayer(hidden_size + self.info_dim, hidden_size)
        self.graph_attention4 = SpGraphAttentionLayer(hidden_size + self.info_dim, hidden_size)
        self.graph_attention5 = SpGraphAttentionLayer(hidden_size + self.info_dim, hidden_size)


    def forward(self, feat, goal_feat, info_feat, adj):
        feat_x = torch.cat([feat, goal_feat, info_feat], dim=-1)
        feat_x = self.feat_enc(feat_x)

        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        feat_x = self.dropout(self.relu(self.graph_attention1(feat_x, adj)))
        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        feat_x = self.dropout(self.relu(self.graph_attention2(feat_x, adj)))
        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        feat_x = self.dropout(self.relu(self.graph_attention3(feat_x, adj)))
        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        feat_x = self.dropout(self.relu(self.graph_attention4(feat_x, adj)))
        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        feat_x = self.dropout(self.relu(self.graph_attention5(feat_x, adj)))
        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist

class TopoGCN_v2_1(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGCN_v2_1, self).__init__()
        self.args = args
        self.info_dim = 1 + 3 + args.vis_feat_dim  # visited, position, goal text feat dim
        if args.use_cm_score:
            self.info_dim += 1  # cm_score
        self.feat_dim = args.vis_feat_dim + self.info_dim

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

        self.feat_enc = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, 1),
            # nn.Sigmoid()
        )

        self.graph_conv1 = GraphConv(hidden_size + self.info_dim, hidden_size)
        self.graph_conv2 = GraphConv(hidden_size + self.info_dim, hidden_size)
        self.graph_conv3 = GraphConv(hidden_size + self.info_dim, hidden_size)
        self.graph_conv4 = GraphConv(hidden_size + self.info_dim, hidden_size)
        self.graph_conv5 = GraphConv(hidden_size + self.info_dim, hidden_size)


    def forward(self, feat, goal_feat, info_feat, adj):
        feat_x = torch.cat([feat, goal_feat, info_feat], dim=-1)
        feat_x = self.feat_enc(feat_x)

        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        feat_x = self.dropout(self.relu(self.graph_conv1(feat_x, adj)))
        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        feat_x = self.dropout(self.relu(self.graph_conv2(feat_x, adj)))
        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        feat_x = self.dropout(self.relu(self.graph_conv3(feat_x, adj)))
        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        feat_x = self.dropout(self.relu(self.graph_conv4(feat_x, adj)))
        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        feat_x = self.dropout(self.relu(self.graph_conv5(feat_x, adj)))

        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist

class TopoGCN_v2_1_depth(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGCN_v2_1_depth, self).__init__()
        self.args = args
        self.info_dim = 1 + 3 + 12 + args.vis_feat_dim  # visited, position, goal text feat dim
        if args.use_cm_score:
            self.info_dim += 1  # cm_score
        self.feat_dim = args.vis_feat_dim + self.info_dim

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

        self.feat_enc = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, 1),
            # nn.Sigmoid()
        )

        self.graph_conv1 = GraphConv(hidden_size + self.info_dim, hidden_size)
        self.graph_conv2 = GraphConv(hidden_size + self.info_dim, hidden_size)
        self.graph_conv3 = GraphConv(hidden_size + self.info_dim, hidden_size)
        self.graph_conv4 = GraphConv(hidden_size + self.info_dim, hidden_size)
        self.graph_conv5 = GraphConv(hidden_size + self.info_dim, hidden_size)


    def forward(self, feat, goal_feat, info_feat, adj):
        feat_x = torch.cat([feat, goal_feat, info_feat], dim=-1)
        feat_x = self.feat_enc(feat_x)

        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        feat_x = self.dropout(self.relu(self.graph_conv1(feat_x, adj)))
        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        feat_x = self.dropout(self.relu(self.graph_conv2(feat_x, adj)))
        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        feat_x = self.dropout(self.relu(self.graph_conv3(feat_x, adj)))
        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        feat_x = self.dropout(self.relu(self.graph_conv4(feat_x, adj)))
        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        feat_x = self.dropout(self.relu(self.graph_conv5(feat_x, adj)))

        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist


class TopoGCN_v1_pano(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGCN_v1_pano, self).__init__()
        self.args = args
        self.info_dim = 1 + 3 + args.vis_feat_dim  # visited, position, goal text feat dim
        if args.use_cm_score:
            self.info_dim += 12 * 5  # cm_score
        self.feat_dim = 12 * args.vis_feat_dim + self.info_dim

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

        self.feat_enc = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, 1),
            # nn.Sigmoid()
        )

        self.graph_conv1 = GraphConv(hidden_size, hidden_size)
        self.graph_conv2 = GraphConv(hidden_size, hidden_size)
        self.graph_conv3 = GraphConv(hidden_size, hidden_size)
        self.graph_conv4 = GraphConv(hidden_size, hidden_size)
        self.graph_conv5 = GraphConv(hidden_size, hidden_size)


    def forward(self, feat, goal_feat, info_feat, adj):
        feat_x = torch.cat([feat, goal_feat, info_feat], dim=-1)
        feat_x = self.feat_enc(feat_x)

        feat_x = self.dropout(self.relu(self.graph_conv1(feat_x, adj)))
        feat_x = self.dropout(self.relu(self.graph_conv2(feat_x, adj)))
        feat_x = self.dropout(self.relu(self.graph_conv3(feat_x, adj)))
        feat_x = self.dropout(self.relu(self.graph_conv4(feat_x, adj)))
        feat_x = self.dropout(self.relu(self.graph_conv5(feat_x, adj)))

        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist


class TopoGCN_v2_2(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGCN_v2_2, self).__init__()
        self.args = args
        self.info_dim = 1 + 3 + args.vis_feat_dim  # visited, position, goal text feat dim
        if args.use_cm_score:
            self.info_dim += 1  # cm_score
        self.feat_dim = args.vis_feat_dim + self.info_dim

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

        self.feat_enc = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, 1),
            # nn.Sigmoid()
        )

        self.graph_conv1 = GraphConv(hidden_size + self.info_dim, hidden_size)
        self.graph_conv2 = GraphConv(hidden_size + self.info_dim, hidden_size)
        self.graph_conv3 = GraphConv(hidden_size + self.info_dim, hidden_size)
        # self.graph_conv4 = GraphConv(hidden_size + self.info_dim, hidden_size)
        # self.graph_conv5 = GraphConv(hidden_size + self.info_dim, hidden_size)


    def forward(self, feat, goal_feat, info_feat, adj):
        feat_x = torch.cat([feat, goal_feat, info_feat], dim=-1)
        feat_x = self.feat_enc(feat_x)

        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        feat_x = self.dropout(self.relu(self.graph_conv1(feat_x, adj)))
        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        feat_x = self.dropout(self.relu(self.graph_conv2(feat_x, adj)))
        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        feat_x = self.dropout(self.relu(self.graph_conv3(feat_x, adj)))
        # feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        # feat_x = self.dropout(self.relu(self.graph_conv4(feat_x, adj)))
        # feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        # feat_x = self.dropout(self.relu(self.graph_conv5(feat_x, adj)))

        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist

class TopoGCN_v2_pano(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGCN_v2_pano, self).__init__()
        self.args = args
        self.info_dim = 1 + 3 + args.vis_feat_dim  # visited, position, goal text feat dim
        if args.use_cm_score:
            self.info_dim += 12 * 5  # cm_score
        self.feat_dim = 12 * args.vis_feat_dim + self.info_dim

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

        self.feat_enc = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, 1),
            # nn.Sigmoid()
        )

        self.graph_conv1 = GraphConv(hidden_size + self.info_dim, hidden_size)
        self.graph_conv2 = GraphConv(hidden_size + self.info_dim, hidden_size)
        self.graph_conv3 = GraphConv(hidden_size + self.info_dim, hidden_size)
        self.graph_conv4 = GraphConv(hidden_size + self.info_dim, hidden_size)
        self.graph_conv5 = GraphConv(hidden_size + self.info_dim, hidden_size)


    def forward(self, feat, goal_feat, info_feat, adj):
        feat_x = torch.cat([feat, goal_feat, info_feat], dim=-1)
        feat_x = self.feat_enc(feat_x)

        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        feat_x = self.dropout(self.relu(self.graph_conv1(feat_x, adj)))
        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        feat_x = self.dropout(self.relu(self.graph_conv2(feat_x, adj)))
        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        feat_x = self.dropout(self.relu(self.graph_conv3(feat_x, adj)))
        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        feat_x = self.dropout(self.relu(self.graph_conv4(feat_x, adj)))
        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        feat_x = self.dropout(self.relu(self.graph_conv5(feat_x, adj)))

        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist




class TopoGCN_v2_pano_goalscore(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGCN_v2_pano_goalscore, self).__init__()
        self.args = args
        self.info_dim = 1 + 3 + args.vis_feat_dim  # visited, position, goal text feat dim
        if args.use_cm_score:
            self.info_dim += 12 * 10  # cm_score
        self.feat_dim = 12 * args.vis_feat_dim + self.info_dim

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

        self.feat_enc = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, 1),
            # nn.Sigmoid()
        )

        self.graph_conv1 = GraphConv(hidden_size + self.info_dim, hidden_size)
        self.graph_conv2 = GraphConv(hidden_size + self.info_dim, hidden_size)
        self.graph_conv3 = GraphConv(hidden_size + self.info_dim, hidden_size)
        self.graph_conv4 = GraphConv(hidden_size + self.info_dim, hidden_size)
        self.graph_conv5 = GraphConv(hidden_size + self.info_dim, hidden_size)


    def forward(self, feat, goal_feat, info_feat, adj):
        feat_x = torch.cat([feat, goal_feat, info_feat], dim=-1)
        feat_x = self.feat_enc(feat_x)

        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        feat_x = self.dropout(self.relu(self.graph_conv1(feat_x, adj)))
        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        feat_x = self.dropout(self.relu(self.graph_conv2(feat_x, adj)))
        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        feat_x = self.dropout(self.relu(self.graph_conv3(feat_x, adj)))
        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        feat_x = self.dropout(self.relu(self.graph_conv4(feat_x, adj)))
        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        feat_x = self.dropout(self.relu(self.graph_conv5(feat_x, adj)))

        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist


class TopoGCN_v3_pano_goalscore(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGCN_v3_pano_goalscore, self).__init__()
        self.args = args
        self.info_dim = 1 + 3 + args.vis_feat_dim  # visited, position, goal text feat dim
        if args.use_cm_score:
            self.info_dim += 12 * 10  # cm_score
        self.feat_dim = 12 * args.vis_feat_dim + self.info_dim

        self.gcn_layer_num = 5

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

        self.feat_enc = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, 1),
            # nn.Sigmoid()
        )

        self.graph_convs = nn.ModuleList()
        for i in range(self.gcn_layer_num):
            self.graph_convs.append(GraphConv(hidden_size + self.info_dim, hidden_size))



    def forward(self, feat, goal_feat, info_feat, adj):
        feat_x = torch.cat([feat, goal_feat, info_feat], dim=-1)
        feat_x = self.feat_enc(feat_x)

        for i in range(self.gcn_layer_num):
            feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
            feat_x = self.dropout(self.relu(self.graph_convs[i](feat_x, adj)))

        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist


class TopoGCN_v3_1_pano_goalscore(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGCN_v3_1_pano_goalscore, self).__init__()
        self.args = args
        self.info_dim = 1 + 3 + args.vis_feat_dim  # visited, position, goal text feat dim
        if args.use_cm_score:
            self.info_dim += 12 * 10  # cm_score
        self.feat_dim = 12 * args.vis_feat_dim + self.info_dim

        self.gcn_layer_num = 5

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

        self.feat_enc = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size + args.vis_feat_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, 1),
            # nn.Sigmoid()
        )

        self.graph_convs = nn.ModuleList()
        for i in range(self.gcn_layer_num):
            self.graph_convs.append(GraphConv(hidden_size + self.info_dim, hidden_size))



    def forward(self, feat, goal_feat, info_feat, adj):
        feat_x = torch.cat([feat, goal_feat, info_feat], dim=-1)
        feat_x = self.feat_enc(feat_x)

        for i in range(self.gcn_layer_num):
            feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
            feat_x = self.dropout(self.relu(self.graph_convs[i](feat_x, adj)))

        feat_x = torch.cat([feat_x, goal_feat], dim=-1)
        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist

class TopoGCN_v3_2_pano_goalscore(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGCN_v3_2_pano_goalscore, self).__init__()
        self.args = args
        self.info_dim = 1 + 3 + args.vis_feat_dim  # visited, position, goal text feat dim
        if args.use_cm_score:
            self.info_dim += 12 * 10  # cm_score
        self.feat_dim = 12 * args.vis_feat_dim + self.info_dim

        self.gcn_layer_num = args.gcn_layers

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.9)
        self.sigmoid = nn.Sigmoid()

        self.feat_enc = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size + args.vis_feat_dim, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, 1),
            # nn.Sigmoid()
        )

        self.graph_convs = nn.ModuleList()
        for i in range(self.gcn_layer_num):
            self.graph_convs.append(GraphConv(hidden_size + self.info_dim, hidden_size))



    def forward(self, feat, goal_feat, info_feat, adj):
        feat_x = torch.cat([feat, goal_feat, info_feat], dim=-1)
        feat_x = self.feat_enc(feat_x)

        for i in range(self.gcn_layer_num):
            feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
            feat_x = self.dropout(self.relu(self.graph_convs[i](feat_x, adj)))

        feat_x = torch.cat([feat_x, goal_feat], dim=-1)
        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist


class TopoGCN_v3_3_pano_goalscore(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGCN_v3_3_pano_goalscore, self).__init__()
        self.args = args
        self.info_dim = 1 + 3 + args.vis_feat_dim  # visited, position, goal text feat dim
        if args.use_cm_score:
            self.info_dim += 12 * 10  # cm_score
        self.feat_dim = 12 * args.vis_feat_dim + self.info_dim

        self.gcn_layer_num = args.gcn_layers

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

        self.feat_enc = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size + self.feat_dim, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, 1),
            # nn.Sigmoid()
        )

        self.graph_convs = nn.ModuleList()
        for i in range(self.gcn_layer_num):
            self.graph_convs.append(GraphConv(hidden_size + self.info_dim, hidden_size))



    def forward(self, feat, goal_feat, info_feat, adj):
        feat_x = torch.cat([feat, goal_feat, info_feat], dim=-1)
        feat_x = self.feat_enc(feat_x)

        for i in range(self.gcn_layer_num):
            feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
            feat_x = self.dropout(self.relu(self.graph_convs[i](feat_x, adj)))

        feat_x = torch.cat([feat_x, goal_feat, info_feat, feat], dim=-1)
        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist

class TopoGCN_v3_4_pano_goalscore(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGCN_v3_4_pano_goalscore, self).__init__()
        self.args = args
        self.info_dim = 1 + 3 + args.vis_feat_dim  # visited, position, goal text feat dim
        if args.use_cm_score:
            self.info_dim += 12 * 10  # cm_score
        self.feat_dim = 12 * args.vis_feat_dim + self.info_dim

        self.gcn_layer_num = args.gcn_layers

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

        self.feat_enc = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size + self.info_dim, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, 1),
            # nn.Sigmoid()
        )

        self.graph_convs = nn.ModuleList()
        for i in range(self.gcn_layer_num):
            self.graph_convs.append(GraphConv(hidden_size + self.info_dim, hidden_size))



    def forward(self, feat, goal_feat, info_feat, adj):
        feat_x = torch.cat([feat, goal_feat, info_feat], dim=-1)
        feat_x = self.feat_enc(feat_x)

        for i in range(self.gcn_layer_num):
            feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
            feat_x = self.dropout(self.relu(self.graph_convs[i](feat_x, adj)))

        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist



class TopoGCN_v4_pano_goalscore(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGCN_v4_pano_goalscore, self).__init__()
        self.args = args
        self.info_dim = 1 + 3 + args.vis_feat_dim  # visited, position, goal text feat dim
        if args.use_cm_score:
            self.info_dim += 12 * 10  # cm_score
        self.feat_dim = 12 * args.vis_feat_dim + self.info_dim

        self.gcn_layer_num = 10

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

        self.feat_enc = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, 1),
            # nn.Sigmoid()
        )

        self.graph_convs = nn.ModuleList()
        for i in range(self.gcn_layer_num):
            self.graph_convs.append(GraphConv(hidden_size, hidden_size))



    def forward(self, feat, goal_feat, info_feat, adj):
        feat_x = torch.cat([feat, goal_feat, info_feat], dim=-1)
        feat_x = self.feat_enc(feat_x)

        for i in range(self.gcn_layer_num):
            feat_x = self.dropout(self.relu(self.graph_convs[i](feat_x, adj)))

        # feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist


class TopoGCN_v4_1_pano_goalscore(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGCN_v4_1_pano_goalscore, self).__init__()
        self.args = args
        self.info_dim = 1 + 3 + args.vis_feat_dim  # visited, position, goal text feat dim
        if args.use_cm_score:
            self.info_dim += 12 * 10  # cm_score
        self.feat_dim = 12 * args.vis_feat_dim + self.info_dim

        self.gcn_layer_num = 10

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

        self.feat_enc = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size + args.vis_feat_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, 1),
            # nn.Sigmoid()
        )

        self.graph_convs = nn.ModuleList()
        for i in range(self.gcn_layer_num):
            self.graph_convs.append(GraphConv(hidden_size, hidden_size))



    def forward(self, feat, goal_feat, info_feat, adj):
        feat_x = torch.cat([feat, goal_feat, info_feat], dim=-1)
        feat_x = self.feat_enc(feat_x)

        for i in range(self.gcn_layer_num):
            feat_x = self.dropout(self.relu(self.graph_convs[i](feat_x, adj)))

        feat_x = torch.cat([feat_x, goal_feat], dim=-1)
        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist

class TopoGCN_v4_2_pano_goalscore(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGCN_v4_2_pano_goalscore, self).__init__()
        self.args = args
        self.info_dim = 1 + 3 + args.vis_feat_dim  # visited, position, goal text feat dim
        if args.use_cm_score:
            self.info_dim += 12 * 10  # cm_score
        self.feat_dim = 12 * args.vis_feat_dim + self.info_dim

        self.gcn_layer_num = args.gcn_layers

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.8)
        self.sigmoid = nn.Sigmoid()

        self.feat_enc = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size + args.vis_feat_dim, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size* 2, 1),
            # nn.Sigmoid()
        )

        self.graph_convs = nn.ModuleList()
        for i in range(self.gcn_layer_num):
            self.graph_convs.append(GraphConv(hidden_size, hidden_size))



    def forward(self, feat, goal_feat, info_feat, adj):
        feat_x = torch.cat([feat, goal_feat, info_feat], dim=-1)
        feat_x = self.feat_enc(feat_x)

        for i in range(self.gcn_layer_num):
            feat_x = self.dropout(self.relu(self.graph_convs[i](feat_x, adj)))

        feat_x = torch.cat([feat_x, goal_feat], dim=-1)
        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist


class TopoGCN_v4_3_pano_goalscore(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGCN_v4_3_pano_goalscore, self).__init__()
        self.args = args
        self.info_dim = 1 + 3 + args.vis_feat_dim  # visited, position, goal text feat dim
        if args.use_cm_score:
            self.info_dim += 12 * 10  # cm_score
        self.feat_dim = 12 * args.vis_feat_dim + self.info_dim

        self.gcn_layer_num = args.gcn_layers

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.8)
        self.sigmoid = nn.Sigmoid()

        self.feat_enc = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size + self.feat_dim, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size* 2, 1),
            # nn.Sigmoid()
        )

        self.graph_convs = nn.ModuleList()
        for i in range(self.gcn_layer_num):
            self.graph_convs.append(GraphConv(hidden_size, hidden_size))



    def forward(self, feat, goal_feat, info_feat, adj):
        feat_x = torch.cat([feat, goal_feat, info_feat], dim=-1)
        feat_x = self.feat_enc(feat_x)

        for i in range(self.gcn_layer_num):
            feat_x = self.dropout(self.relu(self.graph_convs[i](feat_x, adj)))

        feat_x = torch.cat([feat_x, feat, goal_feat, info_feat], dim=-1)
        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist


class TopoGCN_v5_pano_goalscore(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGCN_v5_pano_goalscore, self).__init__()
        self.args = args
        self.info_dim = 1 + 3 + args.vis_feat_dim  # visited, position, goal text feat dim
        if args.use_cm_score:
            self.info_dim += 12 * 10  # cm_score
        self.feat_dim = 12 * args.vis_feat_dim + self.info_dim

        self.gcn_layer_num = args.gcn_layers

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.9)
        self.sigmoid = nn.Sigmoid()

        self.feat_enc = nn.Sequential(
            nn.Linear(12 * args.vis_feat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size + self.info_dim, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size* 2, 1),
            # nn.Sigmoid()
        )

        self.graph_convs = nn.ModuleList()
        for i in range(self.gcn_layer_num):
            self.graph_convs.append(GraphConv(hidden_size, hidden_size))



    def forward(self, feat, goal_feat, info_feat, adj):
        # feat_x = torch.cat([feat, goal_feat, info_feat], dim=-1)
        feat_x = self.feat_enc(feat)

        for i in range(self.gcn_layer_num):
            feat_x = self.dropout(self.relu(self.graph_convs[i](feat_x, adj)))

        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist


class TopoGCN_v5_1_pano_goalscore(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGCN_v5_1_pano_goalscore, self).__init__()
        self.args = args
        self.info_dim = 1 + 3 + args.vis_feat_dim  # visited, position, goal text feat dim
        if args.use_cm_score:
            self.info_dim += 12 * 10  # cm_score
        self.feat_dim = 12 * args.vis_feat_dim + self.info_dim

        self.gcn_layer_num = args.gcn_layers

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.9)
        self.sigmoid = nn.Sigmoid()

        self.feat_enc = nn.Sequential(
            nn.Linear(12 * args.vis_feat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size + self.info_dim, hidden_size*2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size*2, hidden_size*2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size*2, hidden_size*2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size*2, 1),
            # nn.Sigmoid()
        )

        self.graph_convs = nn.ModuleList()
        for i in range(self.gcn_layer_num):
            self.graph_convs.append(GraphConv(hidden_size, hidden_size))



    def forward(self, feat, goal_feat, info_feat, adj):
        # feat_x = torch.cat([feat, goal_feat, info_feat], dim=-1)
        feat_x = self.feat_enc(feat)

        for i in range(self.gcn_layer_num):
            feat_x = self.dropout(self.relu(self.graph_convs[i](feat_x, adj)))

        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist

class TopoGCN_v6_pano_goalscore(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGCN_v6_pano_goalscore, self).__init__()
        self.args = args
        self.info_dim = 1 + 3 + args.vis_feat_dim  # visited, position, goal text feat dim
        if args.use_cm_score:
            self.info_dim += 12 * 10  # cm_score
        self.feat_dim = 12 * args.vis_feat_dim + self.info_dim

        self.gcn_layer_num = args.gcn_layers

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.9)
        self.sigmoid = nn.Sigmoid()

        self.feat_enc = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, 1),
            # nn.Sigmoid()
        )

        self.graph_convs = nn.ModuleList()
        self.node_feat_enc = nn.ModuleList()
        for i in range(self.gcn_layer_num):
            self.graph_convs.append(GraphConv(hidden_size, hidden_size))
            self.node_feat_enc.append(nn.Sequential(
                nn.Linear(hidden_size + self.info_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            ))

    def forward(self, feat, goal_feat, info_feat, adj):
        feat_x = torch.cat([feat, goal_feat, info_feat], dim=-1)
        feat_x = self.feat_enc(feat_x)

        for i in range(self.gcn_layer_num):
            feat_x = self.dropout(self.relu(self.graph_convs[i](feat_x, adj)))
            feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
            feat_x = self.node_feat_enc[i](feat_x)

        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist

class TopoGCN_v7_pano_goalscore(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGCN_v7_pano_goalscore, self).__init__()
        self.args = args
        self.info_dim = 1 + 3 + args.vis_feat_dim  # visited, position, goal text feat dim
        if args.use_cm_score:
            self.info_dim += 12 * 10  # cm_score
        self.feat_dim = 12 * args.vis_feat_dim + self.info_dim

        self.gcn_layer_num = args.gcn_layers

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.9)
        self.sigmoid = nn.Sigmoid()

        self.feat_enc = nn.Sequential(
            nn.Linear(12 * args.vis_feat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size * (self.gcn_layer_num + 1) + self.info_dim, hidden_size*2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size*2, hidden_size*2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size*2, 1),
            # nn.Sigmoid()
        )

        self.graph_convs = nn.ModuleList()
        self.node_feat_enc = nn.ModuleList()
        for i in range(self.gcn_layer_num):
            self.graph_convs.append(GraphConv(hidden_size, hidden_size))
            self.node_feat_enc.append(nn.Sequential(
                nn.Linear(hidden_size + self.info_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            ))



    def forward(self, feat, goal_feat, info_feat, adj):
        feat_x = self.feat_enc(feat)

        feat_list = [feat_x]
        for i in range(self.gcn_layer_num):
            feat_x = self.dropout(self.relu(self.graph_convs[i](feat_x, adj)))
            feat_list.append(feat_x)

        feat_list.append(goal_feat)
        feat_list.append(info_feat)

        feat_x = torch.cat(feat_list, dim=-1)
        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist

class TopoGCN_v7_1_pano_goalscore(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGCN_v7_1_pano_goalscore, self).__init__()
        self.args = args
        self.info_dim = 1 + 3 + args.vis_feat_dim  # visited, position, goal text feat dim
        if args.use_cm_score:
            self.info_dim += 12 * 10  # cm_score
        self.feat_dim = 12 * args.vis_feat_dim + self.info_dim

        self.gcn_layer_num = args.gcn_layers

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.9)
        self.sigmoid = nn.Sigmoid()

        self.feat_enc = nn.Sequential(
            nn.Linear(12 * args.vis_feat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size * (self.gcn_layer_num + 1), hidden_size*2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size*2, hidden_size*2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size*2, 1),
            # nn.Sigmoid()
        )

        self.node_feat_enc0 = nn.Sequential(
            nn.Linear(hidden_size + self.info_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.graph_convs = nn.ModuleList()
        self.node_feat_enc = nn.ModuleList()
        for i in range(self.gcn_layer_num):
            self.graph_convs.append(GraphConv(hidden_size, hidden_size))
            self.node_feat_enc.append(nn.Sequential(
                nn.Linear(hidden_size + self.info_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            ))



    def forward(self, feat, goal_feat, info_feat, adj):
        feat_x = self.feat_enc(feat)

        feat_list = []
        feat_list.append(self.node_feat_enc0(torch.cat([feat_x, goal_feat, info_feat], dim=-1)))
        for i in range(self.gcn_layer_num):
            feat_x = self.dropout(self.relu(self.graph_convs[i](feat_x, adj)))
            feat_list.append(self.node_feat_enc[i](torch.cat([feat_x, goal_feat, info_feat], dim=-1)))

        # feat_list.append(goal_feat)
        # feat_list.append(info_feat)

        feat_x = torch.cat(feat_list, dim=-1)
        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist

class TopoGCN_v7_2_pano_goalscore(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGCN_v7_2_pano_goalscore, self).__init__()
        self.args = args
        self.info_dim = 1 + 3 + args.vis_feat_dim  # visited, position, goal text feat dim
        if args.use_cm_score:
            self.info_dim += 12 * 10  # cm_score
        self.feat_dim = 12 * args.vis_feat_dim + self.info_dim

        self.gcn_layer_num = args.gcn_layers

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

        self.feat_enc = nn.Sequential(
            nn.Linear(12 * args.vis_feat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )



        self.node_feat_enc0 = nn.Sequential(
            nn.Linear(hidden_size + self.info_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.graph_convs = nn.ModuleList()
        self.node_feat_enc = nn.ModuleList()
        for i in range(self.gcn_layer_num):
            self.graph_convs.append(GraphConv(hidden_size, hidden_size))
            self.node_feat_enc.append(nn.Sequential(
                nn.Linear(hidden_size + self.info_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            ))

        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size * (self.gcn_layer_num + 1) + self.info_dim, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 4, 1),
            # nn.Sigmoid()
        )

    def forward(self, feat, goal_feat, info_feat, adj):
        feat_x = self.feat_enc(feat)

        feat_list = []
        feat_list.append(self.node_feat_enc0(torch.cat([feat_x, goal_feat, info_feat], dim=-1)))
        for i in range(self.gcn_layer_num):
            feat_x = self.dropout(self.relu(self.graph_convs[i](feat_x, adj)))
            feat_list.append(self.node_feat_enc[i](torch.cat([feat_x, goal_feat, info_feat], dim=-1)))

        feat_list.append(goal_feat)
        feat_list.append(info_feat)

        feat_x = torch.cat(feat_list, dim=-1)
        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist


class TopoGCN_v7_2_1_pano_goalscore(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGCN_v7_2_1_pano_goalscore, self).__init__()
        self.args = args
        self.info_dim = 1 + 3 + args.vis_feat_dim  # visited, position, goal text feat dim
        if args.use_cm_score:
            self.info_dim += 12 * 10  # cm_score
        self.feat_dim = 12 * args.vis_feat_dim + self.info_dim

        self.gcn_layer_num = args.gcn_layers

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()
        self.batchnorm = nn.BatchNorm1d(hidden_size)

        self.feat_enc = nn.Sequential(
            nn.Linear(12 * args.vis_feat_dim, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )



        self.node_feat_enc0 = nn.Sequential(
            nn.Linear(hidden_size + self.info_dim, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )

        self.graph_convs = nn.ModuleList()
        self.node_feat_enc = nn.ModuleList()
        for i in range(self.gcn_layer_num):
            self.graph_convs.append(GraphConv(hidden_size, hidden_size))
            self.node_feat_enc.append(nn.Sequential(
                nn.Linear(hidden_size + self.info_dim, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, hidden_size),
            ))

        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size * (self.gcn_layer_num + 1) + self.info_dim, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 4, 1),
            # nn.Sigmoid()
        )

    def forward(self, feat, goal_feat, info_feat, adj):
        feat_x = self.feat_enc(feat)

        feat_list = []
        feat_list.append(self.node_feat_enc0(torch.cat([feat_x, goal_feat, info_feat], dim=-1)))
        for i in range(self.gcn_layer_num):
            feat_x = self.batchnorm(self.relu(self.graph_convs[i](feat_x, adj)))
            feat_list.append(self.node_feat_enc[i](torch.cat([feat_x, goal_feat, info_feat], dim=-1)))

        feat_list.append(goal_feat)
        feat_list.append(info_feat)

        feat_x = torch.cat(feat_list, dim=-1)
        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist

class TopoGCN_v7_3_pano_goalscore(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGCN_v7_3_pano_goalscore, self).__init__()
        self.args = args
        self.info_dim = 1 + 3 + args.vis_feat_dim  # visited, position, goal text feat dim
        if args.use_cm_score:
            self.info_dim += 12 * 10  # cm_score
        self.feat_dim = 12 * args.vis_feat_dim + self.info_dim

        self.gcn_layer_num = args.gcn_layers

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

        self.feat_enc = nn.Sequential(
            nn.Linear(12 * args.vis_feat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )



        self.node_feat_enc0 = nn.Sequential(
            nn.Linear(hidden_size + self.info_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.graph_convs = nn.ModuleList()
        self.node_feat_enc = nn.ModuleList()
        for i in range(self.gcn_layer_num):
            self.graph_convs.append(GraphConv(hidden_size, hidden_size))
            self.node_feat_enc.append(nn.Sequential(
                nn.Linear(hidden_size + self.info_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            ))

        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size * (self.gcn_layer_num + 1) + self.info_dim, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 4, 1),
            # nn.Sigmoid()
        )

    def forward(self, feat, goal_feat, info_feat, adj):
        feat_x = self.feat_enc(feat)

        feat_list = []
        feat_x = self.node_feat_enc0(torch.cat([feat_x, goal_feat, info_feat], dim=-1))
        feat_list.append(feat_x)
        for i in range(self.gcn_layer_num):
            feat_x = self.dropout(self.relu(self.graph_convs[i](feat_x, adj)))
            feat_x = self.node_feat_enc[i](torch.cat([feat_x, goal_feat, info_feat], dim=-1))
            feat_list.append(feat_x)

        feat_list.append(goal_feat)
        feat_list.append(info_feat)

        feat_x = torch.cat(feat_list, dim=-1)
        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist

class TopoGCN_v7_3_1_pano_goalscore(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGCN_v7_3_1_pano_goalscore, self).__init__()
        self.args = args
        self.info_dim = 1 + 3 + args.vis_feat_dim  # visited, position, goal text feat dim
        if args.use_cm_score:
            self.info_dim += 12 * 10  # cm_score
        self.feat_dim = 12 * args.vis_feat_dim + self.info_dim

        self.gcn_layer_num = args.gcn_layers

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.batchnorm = nn.BatchNorm1d(hidden_size)
        self.sigmoid = nn.Sigmoid()

        self.feat_enc = nn.Sequential(
            nn.Linear(12 * args.vis_feat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )



        self.node_feat_enc0 = nn.Sequential(
            nn.Linear(hidden_size + self.info_dim, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )

        self.graph_convs = nn.ModuleList()
        self.node_feat_enc = nn.ModuleList()
        for i in range(self.gcn_layer_num):
            self.graph_convs.append(GraphConv(hidden_size, hidden_size))
            self.node_feat_enc.append(nn.Sequential(
                nn.Linear(hidden_size + self.info_dim, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, hidden_size),
            ))

        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size * (self.gcn_layer_num + 1) + self.info_dim, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 4, 1),
            # nn.Sigmoid()
        )

    def forward(self, feat, goal_feat, info_feat, adj):
        feat_x = self.feat_enc(feat)

        feat_list = []
        feat_x = self.node_feat_enc0(torch.cat([feat_x, goal_feat, info_feat], dim=-1))
        feat_list.append(feat_x)
        for i in range(self.gcn_layer_num):
            feat_x = self.batchnorm(self.relu(self.graph_convs[i](feat_x, adj)))
            feat_x = self.node_feat_enc[i](torch.cat([feat_x, goal_feat, info_feat], dim=-1))
            feat_list.append(feat_x)

        feat_list.append(goal_feat)
        feat_list.append(info_feat)

        feat_x = torch.cat(feat_list, dim=-1)
        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist



class TopoGCN_v8_pano_goalscore(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGCN_v8_pano_goalscore, self).__init__()
        self.args = args
        self.info_dim = 1 + 3 + args.vis_feat_dim  # visited, position, goal text feat dim
        if args.cm_type == 'comet':
            self.cm_num = 10
        elif args.cm_type == 'mp3d':
            self.cm_num = 5
        if args.use_cm_score:
            self.info_dim += 12 * self.cm_num  # cm_score
        self.feat_dim = 12 * args.vis_feat_dim + self.info_dim

        self.gcn_layer_num = args.gcn_layers

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()
        self.batchnorm = nn.BatchNorm1d(hidden_size)

        self.feat_enc = nn.Sequential(
            nn.Linear(12 * args.vis_feat_dim, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )



        self.node_feat_enc0 = nn.Sequential(
            nn.Linear(hidden_size + self.info_dim, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )

        self.graph_convs = nn.ModuleList()
        self.node_feat_enc = nn.ModuleList()
        for i in range(self.gcn_layer_num):
            self.graph_convs.append(GraphConv(hidden_size, hidden_size))
            self.node_feat_enc.append(nn.Sequential(
                nn.Linear(hidden_size + self.info_dim, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, hidden_size),
            ))

        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size * (self.gcn_layer_num + 1) + self.info_dim, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 4, 1),
            # nn.Sigmoid()
        )

    def forward(self, feat, goal_feat, info_feat, adj):
        feat_x = self.feat_enc(feat)

        feat_list = []
        feat_list.append(self.node_feat_enc0(torch.cat([feat_x, goal_feat, info_feat], dim=-1)))
        for i in range(self.gcn_layer_num):
            feat_x = self.batchnorm(self.relu(self.graph_convs[i](feat_x, adj))) + feat_x
            feat_list.append(self.node_feat_enc[i](torch.cat([feat_x, goal_feat, info_feat], dim=-1)))

        feat_list.append(goal_feat)
        feat_list.append(info_feat)

        feat_x = torch.cat(feat_list, dim=-1)
        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist

class TopoGCN_v8_1_pano_goalscore(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGCN_v8_1_pano_goalscore, self).__init__()
        self.args = args
        self.info_dim = 1 + 3 + args.vis_feat_dim  # visited, position, goal text feat dim
        if args.use_cm_score:
            self.info_dim += 12 * 10  # cm_score
        self.feat_dim = 12 * args.vis_feat_dim + self.info_dim

        self.gcn_layer_num = args.gcn_layers

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()
        self.batchnorm = nn.BatchNorm1d(hidden_size)

        self.feat_enc = nn.Sequential(
            nn.Linear(12 * args.vis_feat_dim, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )



        self.graph_convs = nn.ModuleList()
        self.node_feat_enc = nn.ModuleList()
        for i in range(self.gcn_layer_num):
            self.graph_convs.append(GraphConv(hidden_size, hidden_size))


        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size + self.info_dim, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 4, 1),
            # nn.Sigmoid()
        )

    def forward(self, feat, goal_feat, info_feat, adj):
        feat_x = self.feat_enc(feat)


        for i in range(self.gcn_layer_num):
            feat_x = self.batchnorm(self.relu(self.graph_convs[i](feat_x, adj))) + feat_x

        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist

class TopoGCN_v8_2_pano_goalscore(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGCN_v8_2_pano_goalscore, self).__init__()
        self.args = args
        self.info_dim = 1 + 3 + args.vis_feat_dim  # visited, position, goal text feat dim
        if args.use_cm_score:
            self.info_dim += 12 * 10  # cm_score
        self.feat_dim = 12 * args.vis_feat_dim + self.info_dim

        self.gcn_layer_num = args.gcn_layers

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()
        self.batchnorm = nn.BatchNorm1d(hidden_size)

        self.feat_enc = nn.Sequential(
            nn.Linear(12 * args.vis_feat_dim, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )



        self.graph_convs = nn.ModuleList()
        self.node_feat_enc = nn.ModuleList()
        for i in range(self.gcn_layer_num):
            self.graph_convs.append(GraphConv(hidden_size + self.info_dim, hidden_size))


        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size + self.info_dim, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 4, 1),
            # nn.Sigmoid()
        )

    def forward(self, feat, goal_feat, info_feat, adj):
        feat_x = self.feat_enc(feat)


        for i in range(self.gcn_layer_num):
            feat_x = self.batchnorm(self.relu(self.graph_convs[i](torch.cat([feat_x, goal_feat, info_feat], dim=-1), adj))) + feat_x

        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist


class TopoGCN_v8_3_pano_goalscore(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGCN_v8_3_pano_goalscore, self).__init__()
        self.args = args
        self.info_dim = 1 + 3 + args.vis_feat_dim  # visited, position, goal text feat dim
        if args.use_cm_score:
            self.info_dim += 12 * 10  # cm_score
        self.feat_dim = 12 * args.vis_feat_dim + self.info_dim

        self.gcn_layer_num = args.gcn_layers

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()
        self.batchnorm = nn.BatchNorm1d(hidden_size)

        self.feat_enc = nn.Sequential(
            nn.Linear(12 * args.vis_feat_dim, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )

        self.graph_convs = nn.ModuleList()
        self.node_feat_enc = nn.ModuleList()
        for i in range(self.gcn_layer_num):
            self.graph_convs.append(GraphConv(hidden_size, hidden_size))

        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size * (self.gcn_layer_num + 1) + self.info_dim, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 4, 1),
            # nn.Sigmoid()
        )

    def forward(self, feat, goal_feat, info_feat, adj):
        feat_x = self.feat_enc(feat)

        feat_list = []
        feat_list.append(feat_x)
        for i in range(self.gcn_layer_num):
            feat_x = self.batchnorm(self.relu(self.graph_convs[i](feat_x, adj))) + feat_x
            feat_list.append(feat_x)

        feat_list.append(goal_feat)
        feat_list.append(info_feat)

        feat_x = torch.cat(feat_list, dim=-1)
        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist


class TopoGAT_v8_pano_goalscore(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGAT_v8_pano_goalscore, self).__init__()
        self.args = args
        self.info_dim = 1 + 3 + args.vis_feat_dim  # visited, position, goal text feat dim
        if args.cm_type == 'comet':
            self.cm_num = 10
        elif args.cm_type == 'mp3d':
            self.cm_num = 5
        if args.use_cm_score:
            self.info_dim += 12 * self.cm_num  # cm_score
        self.feat_dim = 12 * args.vis_feat_dim + self.info_dim

        self.gcn_layer_num = args.gcn_layers

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()
        self.batchnorm = nn.BatchNorm1d(hidden_size)

        self.feat_enc = nn.Sequential(
            nn.Linear(12 * args.vis_feat_dim, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )



        self.node_feat_enc0 = nn.Sequential(
            nn.Linear(hidden_size + self.info_dim, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )

        self.graph_attentions = nn.ModuleList()
        self.node_feat_enc = nn.ModuleList()
        for i in range(self.gcn_layer_num):
            self.graph_attentions.append(GraphAttention(hidden_size, hidden_size))
            self.node_feat_enc.append(nn.Sequential(
                nn.Linear(hidden_size + self.info_dim, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, hidden_size),
            ))

        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size * (self.gcn_layer_num + 1) + self.info_dim, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 4, 1),
            # nn.Sigmoid()
        )

    def forward(self, feat, goal_feat, info_feat, adj):
        feat_x = self.feat_enc(feat)

        feat_list = []
        feat_list.append(self.node_feat_enc0(torch.cat([feat_x, goal_feat, info_feat], dim=-1)))
        for i in range(self.gcn_layer_num):
            feat_x = self.batchnorm(self.relu(self.graph_attentions[i](feat_x, adj))) + feat_x
            feat_list.append(self.node_feat_enc[i](torch.cat([feat_x, goal_feat, info_feat], dim=-1)))

        feat_list.append(goal_feat)
        feat_list.append(info_feat)

        feat_x = torch.cat(feat_list, dim=-1)
        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist


class TopoGAT_v8_1_pano_goalscore(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGAT_v8_1_pano_goalscore, self).__init__()
        self.args = args
        self.info_dim = 1 + 3 + args.vis_feat_dim  # visited, position, goal text feat dim
        if args.use_cm_score:
            self.info_dim += 12 * 10  # cm_score
        self.feat_dim = 12 * args.vis_feat_dim + self.info_dim

        self.gcn_layer_num = args.gcn_layers

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()
        self.batchnorm = nn.BatchNorm1d(hidden_size)

        self.feat_enc = nn.Sequential(
            nn.Linear(12 * args.vis_feat_dim, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )



        self.graph_convs = nn.ModuleList()
        self.node_feat_enc = nn.ModuleList()
        for i in range(self.gcn_layer_num):
            self.graph_convs.append(GraphAttention(hidden_size, hidden_size))


        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size + self.info_dim, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 4, 1),
            # nn.Sigmoid()
        )

    def forward(self, feat, goal_feat, info_feat, adj):
        feat_x = self.feat_enc(feat)


        for i in range(self.gcn_layer_num):
            feat_x = self.batchnorm(self.relu(self.graph_convs[i](feat_x, adj))) + feat_x

        feat_x = torch.cat([feat_x, goal_feat, info_feat], dim=-1)
        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist

class TopoGAT_v9_pano_goalscore(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGAT_v9_pano_goalscore, self).__init__()
        self.args = args
        self.info_dim = 1 + args.vis_feat_dim  # visited, goal text feat dim
        if args.use_cm_score:
            self.info_dim += 12 * 10  # cm_score
        self.feat_dim = 12 * args.vis_feat_dim + self.info_dim

        self.gcn_layer_num = args.gcn_layers

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()
        self.batchnorm = nn.BatchNorm1d(hidden_size)

        self.feat_enc = nn.Sequential(
            nn.Linear(12 * args.vis_feat_dim + self.info_dim, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )



        self.graph_convs = nn.ModuleList()
        self.node_feat_enc = nn.ModuleList()
        for i in range(self.gcn_layer_num):
            self.graph_convs.append(GraphAttention(hidden_size, hidden_size))
            self.node_feat_enc.append(nn.Sequential(
                nn.Linear(hidden_size + self.info_dim, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, hidden_size),
            ))


        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size * (self.gcn_layer_num + 1), hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            # nn.Linear(hidden_size * 2, hidden_size * 4),
            # nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, 1),
            # nn.Sigmoid()
        )

    def forward(self, feat, goal_feat, info_feat, adj):

        # info_feat = torch.cat([info_feat[:, :1], info_feat[:, 4:]], dim=-1) ## remove position info
        feat_x = self.feat_enc(torch.cat([feat, goal_feat, info_feat], dim=-1))

        feat_list = []
        feat_list.append(feat_x)
        for i in range(self.gcn_layer_num):
            feat_x = self.batchnorm(self.relu(self.graph_convs[i](feat_x, adj))) + feat_x
            feat_x = self.node_feat_enc[i](torch.cat([feat_x, goal_feat, info_feat], dim=-1))
            feat_list.append(feat_x)

        feat_x = torch.cat(feat_list, dim=-1)
        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist

class TopoGAT_v9_1_pano_goalscore(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGAT_v9_1_pano_goalscore, self).__init__()
        self.args = args
        self.info_dim = 1 + args.vis_feat_dim  # visited, goal text feat dim
        if args.use_cm_score:
            self.info_dim += 12 * 10  # cm_score
        self.feat_dim = 12 * args.vis_feat_dim + self.info_dim

        self.gcn_layer_num = args.gcn_layers

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()
        self.batchnorm = nn.BatchNorm1d(hidden_size)

        self.feat_enc = nn.Sequential(
            nn.Linear(12 * args.vis_feat_dim + self.info_dim, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )



        self.graph_convs = nn.ModuleList()
        self.node_feat_enc = nn.ModuleList()
        for i in range(self.gcn_layer_num):
            self.graph_convs.append(GraphAttention(hidden_size, hidden_size))
            self.node_feat_enc.append(nn.Sequential(
                nn.Linear(hidden_size + self.info_dim, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, hidden_size),
            ))


        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size * (self.gcn_layer_num + 1), hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 4, 1),
            # nn.Sigmoid()
        )

    def forward(self, feat, goal_feat, info_feat, adj):

        # info_feat = torch.cat([info_feat[:, :1], info_feat[:, 4:]], dim=-1) ## remove position info
        feat_x = self.feat_enc(torch.cat([feat, goal_feat, info_feat], dim=-1))

        feat_list = []
        feat_list.append(feat_x)
        for i in range(self.gcn_layer_num):
            feat_x = self.batchnorm(self.relu(self.graph_convs[i](feat_x, adj))) + feat_x
            feat_list.append(self.node_feat_enc[i](torch.cat([feat_x, goal_feat, info_feat], dim=-1)))

        feat_x = torch.cat(feat_list, dim=-1)
        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist



class TopoGCN_v10_pano_goalscore(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGCN_v10_pano_goalscore, self).__init__()
        self.args = args
        self.info_dim = 1 + 3 + args.vis_feat_dim  # visited, position, goal text feat dim
        if args.cm_type == 'comet':
            self.cm_num = 10
        elif args.cm_type == 'mp3d':
            self.cm_num = 5
        if args.use_cm_score:
            self.info_dim += 12 * self.cm_num  # cm_score
        self.feat_dim = 12 * args.vis_feat_dim + self.info_dim

        self.gcn_layer_num = args.gcn_layers

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()
        self.batchnorm = nn.BatchNorm1d(hidden_size)

        self.vis_feat_enc = nn.Sequential(
            nn.Linear(12 * args.vis_feat_dim, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )

        self.visit_enc = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size)
        )
        self.pos_enc = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size)
        )
        self.goal_enc = nn.Sequential(
            nn.Linear(args.vis_feat_dim, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size)
        )

        if args.use_cm_score:
            self.cm_enc = nn.Sequential(
                nn.Linear(12 * self.cm_num, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, hidden_size)
            )

        input_feat_dim = hidden_size * 4
        if args.use_cm_score:
            input_feat_dim += hidden_size
        self.feat_enc = nn.Sequential(
            nn.Linear(input_feat_dim, hidden_size*2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, hidden_size),
        )


        self.node_feat_enc0 = nn.Sequential(
            nn.Linear(hidden_size + self.info_dim, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )

        self.graph_convs = nn.ModuleList()
        self.node_feat_enc = nn.ModuleList()
        for i in range(self.gcn_layer_num):
            self.graph_convs.append(GraphConv(hidden_size, hidden_size))
            self.node_feat_enc.append(nn.Sequential(
                nn.Linear(hidden_size + hidden_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, hidden_size),
            ))

        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size * (self.gcn_layer_num + 1), hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 4, 1),
            # nn.Sigmoid()
        )

    def forward(self, feat, goal_feat, info_feat, adj):
        feat_x = self.vis_feat_enc(feat)
        goal_feat_x = self.goal_enc(goal_feat)
        visit_feat_x = self.visit_enc(info_feat[:, 0:1])
        pos_feat_x = self.pos_enc(info_feat[:, 1:4])

        if self.args.use_cm_score:
            cm_feat_x = self.cm_enc(info_feat[:, 4:])
            info_feat = torch.cat([goal_feat_x, visit_feat_x, pos_feat_x, cm_feat_x], dim=-1)

        else:
            info_feat = torch.cat([goal_feat_x, visit_feat_x, pos_feat_x], dim=-1)

        feat_list = []
        feat_x = self.feat_enc(torch.cat([feat_x, info_feat], dim=-1))
        feat_list.append(feat_x)
        for i in range(self.gcn_layer_num):
            feat_x = self.batchnorm(self.relu(self.graph_convs[i](feat_x, adj))) + feat_x
            feat_x = self.feat_enc(torch.cat([feat_x, info_feat], dim=-1))
            feat_list.append(feat_x)


        feat_x = torch.cat(feat_list, dim=-1)
        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist


class TopoGCN_v10_1_pano_goalscore(nn.Module):
    def __init__(self, args, hidden_size=512):
        super(TopoGCN_v10_1_pano_goalscore, self).__init__()
        self.args = args
        self.info_dim = 1 + 3 + args.vis_feat_dim  # visited, position, goal text feat dim
        if args.cm_type == 'comet':
            self.cm_num = 10
        elif args.cm_type == 'mp3d':
            self.cm_num = 5
        if args.use_cm_score:
            self.info_dim += 12 * self.cm_num  # cm_score
        self.feat_dim = 12 * args.vis_feat_dim + self.info_dim

        self.gcn_layer_num = args.gcn_layers

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()
        self.batchnorm = nn.BatchNorm1d(hidden_size)

        self.vis_feat_enc = nn.Sequential(
            nn.Linear(12 * args.vis_feat_dim, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )

        self.visit_enc = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 32)
        )
        self.pos_enc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64)
        )
        self.goal_enc = nn.Sequential(
            nn.Linear(args.vis_feat_dim, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size)
        )

        if args.use_cm_score:
            self.cm_enc = nn.Sequential(
                nn.Linear(12 * self.cm_num, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, hidden_size)
            )

        input_feat_dim = hidden_size + 32 + 64 + hidden_size
        if args.use_cm_score:
            input_feat_dim += hidden_size
        self.feat_enc0 = nn.Sequential(
            nn.Linear(input_feat_dim, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )

        self.graph_convs = nn.ModuleList()
        self.node_feat_enc = nn.ModuleList()
        for i in range(self.gcn_layer_num):
            self.graph_convs.append(GraphConv(hidden_size, hidden_size))
            self.node_feat_enc.append(nn.Sequential(
                nn.Linear(input_feat_dim, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, hidden_size),
            ))

        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size * (self.gcn_layer_num + 1) + (input_feat_dim - hidden_size), hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 4, 1),
            # nn.Sigmoid()
        )

    def forward(self, feat, goal_feat, info_feat, adj):
        feat_x = self.vis_feat_enc(feat)
        goal_feat_x = self.goal_enc(goal_feat)
        visit_feat_x = self.visit_enc(info_feat[:, 0:1])
        pos_feat_x = self.pos_enc(info_feat[:, 1:4])

        if self.args.use_cm_score:
            cm_feat_x = self.cm_enc(info_feat[:, 4:])
            info_feat = torch.cat([goal_feat_x, visit_feat_x, pos_feat_x, cm_feat_x], dim=-1)

        else:
            info_feat = torch.cat([goal_feat_x, visit_feat_x, pos_feat_x], dim=-1)

        feat_list = []
        feat_list.append(self.feat_enc0(torch.cat([feat_x, info_feat], dim=-1)))
        for i in range(self.gcn_layer_num):
            feat_x = self.batchnorm(self.relu(self.graph_convs[i](feat_x, adj))) + feat_x
            feat_list.append(self.node_feat_enc[i](torch.cat([feat_x, info_feat], dim=-1)))

        feat_list.append(info_feat)

        feat_x = torch.cat(feat_list, dim=-1)
        pred_dist = self.sigmoid(self.value_layer(feat_x))

        return pred_dist