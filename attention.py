import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from fc import FCNet
from bc import BCNet

from multi_layer_net import MultiLayerNet


def get_norm(norm):
    no_norm = lambda x, dim: x
    if norm == 'weight':
        norm_layer = weight_norm
    elif norm == 'batch':
        norm_layer = nn.BatchNorm1d
    elif norm == 'layer':
        norm_layer = nn.LayerNorm
    elif norm == 'none':
        norm_layer = no_norm
    else:
        print("Invalid Normalization")
        raise Exception("Invalid Normalization")
    return norm_layer



class Attention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid):
        super(Attention, self).__init__()
        self.nonlinear = FCNet([v_dim + q_dim, num_hid])
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        vq = torch.cat((v, q), 2)
        joint_repr = self.nonlinear(vq)
        logits = self.linear(joint_repr)
        return logits


class NewAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(NewAttention, self).__init__()

        self.v_proj = FCNet([v_dim, num_hid])
        self.q_proj = FCNet([q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        #self.linear = weight_norm(nn.Linear(q_dim, 1), dim=None)
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)
    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v) # [batch, k, qdim]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits

class BiAttention(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, glimpse, dropout=[.2,.5]):
        super(BiAttention, self).__init__()

        self.glimpse = glimpse
        self.logits = weight_norm(BCNet(x_dim, y_dim, z_dim, glimpse, dropout=dropout, k=3), \
            name='h_mat', dim=None)

    def forward(self, v, q, v_mask=True):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        p, logits = self.forward_all(v, q, v_mask)
        return p, logits

    def forward_all(self, v, q, v_mask=True, logit=False, mask_with=-float('inf')):
        v_num = v.size(1)
        q_num = q.size(1)
        logits = self.logits(v,q) # b x g x v x q

        if v_mask:
            mask = (0 == v.abs().sum(2)).unsqueeze(1).unsqueeze(3).expand(logits.size())
            logits.data.masked_fill_(mask.data, mask_with)

        if not logit:
            p = nn.functional.softmax(logits.view(-1, self.glimpse, v_num * q_num), 2)
            return p.view(-1, self.glimpse, v_num, q_num), logits

        return logits
    
class Attention2(nn.Module):
    def __init__(self, input_dimension, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # self.att_w_1 = MultiLayerNet(
        #     [input_dimension, input_dimension],
        #     dropout=dropout,
        #     activation_fn_name=None,
        # )
        # self.att_w_2 = MultiLayerNet(
        #     [input_dimension, 1], activation_fn_name=None
        # )
        self.linear = weight_norm(nn.Linear(input_dimension, 1), dim=None)

    def forward(self, inp):
        logits = self.logits(inp)
        return nn.functional.softmax(logits, dim=1)

    def logits(self, inp):
        similarity_matrix = self.dropout(inp)
        # w_1 = torch.tanh(self.att_w_1(similarity_matrix))
        # return self.att_w_2(w_1)
        logits = self.linear(similarity_matrix)
        return logits    
    
class SelfAttention(nn.Module):
    def __init__(self, input_dimension, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.self_att_w_1 = MultiLayerNet(
            [input_dimension, input_dimension],
            dropout=dropout,
            activation_fn_name=None,
        )
        self.self_att_w_2 = MultiLayerNet(
            [input_dimension, 1], activation_fn_name=None
        )

    def forward(self, inp):
        logits = self.logits(inp)
        return nn.functional.softmax(logits, dim=1)

    def logits(self, inp):
        inp = self.dropout(inp)
        w_1 = torch.tanh(self.self_att_w_1(inp))
        return self.self_att_w_2(w_1)


class ReAttention(nn.Module):
    def __init__(
        self, hidden_dimension, number_of_objects, dropout=0.2
    ):
        super().__init__()
        self.number_of_objects = number_of_objects

        self.non_linear_layer = None

        self.dropout = nn.Dropout(dropout)

        self.linear = weight_norm(nn.Linear(hidden_dimension, 1), dim=None)

    def forward(self, r, q_proj, v_proj):
        answer_representation = r.unsqueeze(1).repeat(
            1, self.number_of_objects, 1
        )

        joint_repr = answer_representation * v_proj

        if self.non_linear_layer:
            joint_repr = self.non_linear_layer(joint_repr)

        joint_repr = self.dropout(joint_repr)

        return nn.functional.softmax(self.linear(joint_repr), dim=1)
    
    
# 1 layer seperate, element-wise *, 1 layer seperate, output layer
class Att_3(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm, act, dropout=0.0):
        super(Att_3, self).__init__()
        norm_layer = get_norm(norm)
        self.v_proj = FCNet([v_dim, num_hid], dropout= dropout, act= act)
        self.q_proj = FCNet([q_dim, num_hid], dropout= dropout, act= act)
        self.nonlinear = FCNet([num_hid, num_hid], dropout= dropout, act= act)
        self.linear = norm_layer(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v) # [batch, k, num_hid]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1) # [batch, k, num_hid]
        joint_repr = v_proj * q_proj
        joint_repr = self.nonlinear(joint_repr)
        logits = self.linear(joint_repr)
        return logits
    
    



class GAttNet(nn.Module):
    def __init__(self, dir_num, label_num, in_feat_dim, out_feat_dim,
                 nongt_dim=20, dropout=0.2, label_bias=True, 
                 num_heads=16, pos_emb_dim=-1):
        """ Attetion module with vectorized version

        Args:
            label_num: numer of edge labels
            dir_num: number of edge directions
            feat_dim: dimension of roi_feat
            pos_emb_dim: dimension of postion embedding for implicit relation, set as -1 for explicit relation

        Returns:
            output: [num_rois, ovr_feat_dim, output_dim]
        """
        super(GAttNet, self).__init__()
        assert dir_num <= 2, "Got more than two directions in a graph."
        self.dir_num = dir_num
        self.label_num = label_num
        self.in_feat_dim = in_feat_dim
        self.out_feat_dim = out_feat_dim
        self.dropout = nn.Dropout(dropout)
        self.self_weights = FCNet([in_feat_dim, out_feat_dim], '', dropout)
        self.bias = FCNet([label_num, 1], '', 0, label_bias)
        self.nongt_dim = nongt_dim
        self.pos_emb_dim = pos_emb_dim
        neighbor_net = []
        for i in range(dir_num):
            g_att_layer = GraphSelfAttentionLayer(
                                pos_emb_dim=pos_emb_dim,
                                num_heads=num_heads,
                                feat_dim=out_feat_dim,
                                nongt_dim=nongt_dim)
            neighbor_net.append(g_att_layer)
        self.neighbor_net = nn.ModuleList(neighbor_net)

    def forward(self, v_feat, adj_matrix, pos_emb=None):
        """
        Args:
            v_feat: [batch_size,num_rois, feat_dim]
            adj_matrix: [batch_size, num_rois, num_rois, num_labels]
            pos_emb: [batch_size, num_rois, pos_emb_dim]

        Returns:
            output: [batch_size, num_rois, feat_dim]
        """
        if self.pos_emb_dim > 0 and pos_emb is None:
            raise ValueError(
                f"position embedding is set to None "
                f"with pos_emb_dim {self.pos_emb_dim}")
        elif self.pos_emb_dim < 0 and pos_emb is not None:
            raise ValueError(
                f"position embedding is NOT None "
                f"with pos_emb_dim < 0")
        batch_size, num_rois, feat_dim = v_feat.shape
        nongt_dim = self.nongt_dim

        adj_matrix = adj_matrix.float()

        adj_matrix_list = [adj_matrix, adj_matrix.transpose(1, 2)]

        # Self - looping edges
        # [batch_size,num_rois, out_feat_dim]
        self_feat = self.self_weights(v_feat)

        output = self_feat
        neighbor_emb = [0] * self.dir_num
        for d in range(self.dir_num):
            # [batch_size,num_rois, nongt_dim,label_num]
            input_adj_matrix = adj_matrix_list[d][:, :, :nongt_dim, :]
            condensed_adj_matrix = torch.sum(input_adj_matrix, dim=-1)

            # [batch_size,num_rois, nongt_dim]
            v_biases_neighbors = self.bias(input_adj_matrix).squeeze(-1)

            # [batch_size,num_rois, out_feat_dim]
            neighbor_emb[d] = self.neighbor_net[d].forward(
                        self_feat, condensed_adj_matrix, pos_emb,
                        v_biases_neighbors)

            # [batch_size,num_rois, out_feat_dim]
            output = output + neighbor_emb[d]
        output = self.dropout(output)
        output = nn.functional.relu(output)

        return output
