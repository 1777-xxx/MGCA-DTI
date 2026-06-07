import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout

import random
from tqdm import tqdm
import numpy as np

import dgl
from dgl import DGLGraph

from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.gt import GraphormerLayer
from dgllife.model import GCN, GAT
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch.conv import GATv2Conv
from torch.nn.utils.parametrizations import weight_norm
from dgl.nn.pytorch.glob import GlobalAttentionPooling
from dgl.nn.pytorch import SAGEConv
from dgl.nn.pytorch import SAGEConv, GINConv
# from dgl.nn.pytorch.glob import MeanPooling
from torch.nn import LayerNorm, Dropout, Linear, Sequential, ReLU, BatchNorm1d

from dgl.nn.pytorch import HeteroGraphConv

from torch.nn.init import xavier_uniform_

class AdaptiveGAT(nn.Module):
    def __init__(
            self,
            in_feats,
            hidden_feats=128,  # 固定为128，匹配proj层输入
            n_layers=2,
            n_heads=4,
            dropout=0.1,
            negative_slope=0.2,
            edge_types=None,
            readout='tanh'
    ):
        super().__init__()

        # 核心参数
        self.in_feats = in_feats
        self.hidden_feats = hidden_feats  # 输出维度固定128
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.edge_types = edge_types if edge_types is not None else ['_E']

        # 1. 输入投影：in_feats → 128（强制对齐）
        self.input_proj = nn.Linear(in_feats, hidden_feats)

        # 2. 多层GATConv（兼容同构图/异构图）
        self.gat_layers = nn.ModuleList()
        for layer in range(n_layers):
            # 每层输入维度：第0层=128，后续层=128*4=512
            in_dim = hidden_feats if layer == 0 else hidden_feats * n_heads
            conv_dict = {}
            for etype in self.edge_types:
                conv_dict[etype] = GATConv(
                    in_feats=in_dim,
                    out_feats=hidden_feats,  # 每头输出128维
                    num_heads=n_heads,
                    feat_drop=dropout,
                    attn_drop=dropout,
                    negative_slope=negative_slope,
                    residual=True,
                    activation=F.elu if layer < n_layers - 1 else None
                )
            self.gat_layers.append(HeteroGraphConv(conv_dict, aggregate='mean'))

        # 3. 多头融合层：128*4=512 → 128（核心：强制降维到128）
        self.head_fusion = nn.Sequential(
            nn.Linear(hidden_feats * n_heads, hidden_feats),
            nn.LayerNorm(hidden_feats),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 4. 节点重要性计算函数

    def compute_node_importance(self, g, etype=None):
        if etype is None:
            etype = '_E' if g.is_homogeneous else g.canonical_etypes[0]
        degrees = g.in_degrees(etype=etype).float()
        importance = degrees / (degrees.sum() + 1e-7)  # 防止除零
        return importance

    def forward(self, g, node_feats, node_index=None, etype=None):
        """
        保证最终输出维度：
        - 单样本：(128,) → 堆叠后：(batch_size, 128)
        """
        # ------------- 1. 输入校验与补维 -------------
        if node_feats.dim() == 0:
            raise ValueError(f"AdaptiveGAT调用错误！请传入节点特征g.ndata['h']，当前是0维标量")
        # 确保节点特征是2维 (num_nodes, in_feats)
        if node_feats.dim() == 1:
            node_feats = node_feats.unsqueeze(1)

        # ------------- 2. 输入投影 + 节点重要性加权 -------------
        h = self.input_proj(node_feats)  # (num_nodes, in_feats) → (num_nodes, 128)
        importance = self.compute_node_importance(g)  # (num_nodes,)
        h = h * importance.unsqueeze(1)  # (num_nodes, 128)

        # ------------- 3. 多层GAT编码 -------------
        for layer_idx, gat_layer in enumerate(self.gat_layers):
            # 适配同构图/异构图的输入格式
            if g.is_homogeneous:
                h_dict = {g.ntypes[0]: h}
            else:
                h_dict = {ntype: h[:g.number_of_nodes(ntype)] for ntype in g.ntypes}

            # GAT层前向
            new_h_dict = gat_layer(g, h_dict)
            # 提取输出（取第一个节点类型）
            h = new_h_dict[list(g.ntypes)[0]]
            # 多头展平：(num_nodes, n_heads, 128) → (num_nodes, 128*4=512)
            h = h.flatten(1)

        # ------------- 4. 多头融合 + 强制128维 -------------
        h = self.head_fusion(h)  # (num_nodes, 512) → (num_nodes, 128)

        # ------------- 5. 提取指定节点特征 -------------
        if node_index is not None:
            # 兼容标量/张量索引，确保输出是(128,)
            if isinstance(node_index, int):
                h = h[node_index]
            else:
                # 处理批量索引，保证维度为(128,)
                h = h[node_index].squeeze()

        # 最终输出维度：(128,) → 堆叠后batch维度为 (batch_size, 128)
        return h

    def get_graph_embedding(self, g, node_feats, etype='interacts'):

        node_emb = self.forward(g, node_feats, etype=etype)
        graph_emb = self.readout(node_emb)
        return graph_emb



class DoubleCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attn1 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.cross_attn2 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, drug_feat, prot_feat):
        # drug_feat: (B, 1, D), prot_feat: (B, 1, D)
        # 第一层：drug query -> prot key/value
        attn1, _ = self.cross_attn1(drug_feat, prot_feat, prot_feat)
        drug_feat = self.norm1(drug_feat + self.dropout(attn1))

        # 第二层：prot query -> drug key/value
        attn2, _ = self.cross_attn2(prot_feat, drug_feat, drug_feat)
        prot_feat = self.norm2(prot_feat + self.dropout(attn2))

        return drug_feat, prot_feat

def augment_graph(g, drop_edge_rate=0.1, drop_node_rate=0.05, feat_perturb_rate=0.01):
    if not g.is_homogeneous or g.num_edges() == 0:
        return g

    # 自适应调整增强强度
    num_nodes = g.num_nodes()
    num_edges = g.num_edges()
    adapt_drop_edge = drop_edge_rate * min(1.0, num_edges / 100)
    adapt_feat_perturb = feat_perturb_rate * min(1.0, num_nodes / 50)

    # 1. 随机删边
    if adapt_drop_edge > 0:
        eids = np.arange(num_edges)
        keep_eids = np.random.choice(
            eids,
            size=max(1, int(num_edges * (1 - adapt_drop_edge))),
            replace=False
        )
        g = dgl.edge_subgraph(g, keep_eids)

    # 2. 节点特征扰动
    if adapt_feat_perturb > 0 and 'h' in g.ndata:
        h = g.ndata['h']
        noise = torch.randn_like(h) * adapt_feat_perturb
        perturb_mask = torch.rand(h.shape[0]) < drop_node_rate
        h[perturb_mask] += noise[perturb_mask]
        g.ndata['h'] = h

    return g


class CustomGraphormerAttention(nn.Module):
    """
    自定义Graphormer风格注意力层：包含节点度位置编码+多头自注意力+前馈网络
    核心逻辑与Graphormer一致，但无DGL版本依赖
    """

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # 多头注意力
        self.q_proj = Linear(hidden_dim, hidden_dim)
        self.k_proj = Linear(hidden_dim, hidden_dim)
        self.v_proj = Linear(hidden_dim, hidden_dim)
        self.out_proj = Linear(hidden_dim, hidden_dim)

        # 前馈网络（Graphormer标配）
        self.ffn = Sequential(
            Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            Dropout(dropout),
            Linear(hidden_dim * 4, hidden_dim),
            Dropout(dropout)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # 节点度位置编码（Graphormer核心特征）
        self.pos_enc = nn.Embedding(1000, hidden_dim)

    def forward(self, g, x):
        """
        Args:
            g: DGLGraph - 输入图
            x: (num_nodes, hidden_dim) - 节点特征
        Returns:
            x: (num_nodes, hidden_dim) - 增强后的节点特征
        """
        # 1. 节点度位置编码
        node_deg = g.in_degrees().clamp(0, 999).to(x.device)
        pos_emb = self.pos_enc(node_deg)
        x = x + pos_emb  # 残差连接

        # 2. 多头自注意力
        batch_size = 1
        x_residual = x
        x = x.unsqueeze(0)  # (1, num_nodes, hidden_dim)

        # 投影到q/k/v
        q = self.q_proj(x).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力计算
        attn_scores = (q @ k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = (attn_weights @ v).transpose(1, 2).reshape(batch_size, -1, self.hidden_dim)

        # 输出投影+残差+归一化
        x = self.norm1(x_residual.unsqueeze(0) + self.dropout(self.out_proj(attn_output)))
        x = x.squeeze(0)

        # 3. 前馈网络+残差+归一化
        x_residual2 = x
        x = self.norm2(x_residual2 + self.dropout(self.ffn(x)))

        return x


############################################
# 3. Graphormer增强的GIN模型（使用自定义注意力层）
############################################
class GraphormerGIN(nn.Module):
    def __init__(self, dim_h, num_node_features, num_heads=4, dropout=0.1):
        super().__init__()
        # 原始GIN卷积层
        self.conv1 = GINConv(
            Sequential(
                Linear(num_node_features, dim_h),
                BatchNorm1d(dim_h),
                ReLU(),
                Dropout(dropout)
            )
        )
        self.conv2 = GINConv(
            Sequential(
                Linear(dim_h, dim_h),
                BatchNorm1d(dim_h),
                ReLU(),
                Linear(dim_h, dim_h),
                ReLU(),
                Dropout(dropout)
            )
        )

        # 使用自定义Graphormer注意力层（无版本依赖）
        self.graphormer = CustomGraphormerAttention(
            hidden_dim=dim_h,
            num_heads=num_heads,
            dropout=dropout
        )

    def forward(self, g, h):
        # 1. 基础GIN编码
        h1 = self.conv1(g, h)
        h2 = self.conv2(g, h1)

        # 2. 自定义Graphormer增强
        h2 = self.graphormer(g, h2)
        # h2 = self.conv2(g, h1)

        return h2

class HardNegativeContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, hard_ratio=0.2):
        super().__init__()
        self.temperature = temperature
        self.hard_ratio = hard_ratio  # 难负样本比例

    def forward(self, z_A, z_B, index, cross_index=None):
        if index is None:
            return torch.tensor(0.0, device=z_A.device)

        z_A = F.normalize(z_A, dim=1)
        z_B = F.normalize(z_B, dim=1)

        # print(z_A.shape, z_B.shape)

        # 1. 原始正样本计算
        pos_sim = torch.sum(z_A * z_B, dim=1) / self.temperature
        pos_sim_exp = torch.exp(pos_sim)

        # 2. 难负样本挖掘（选相似度最高的负样本）
        index = index.view(-1, 1)
        neg_mask = (index != index.T).float()

        # A模态内相似度矩阵
        sim_matrix_A = torch.matmul(z_A, z_A.T) / self.temperature
        # 屏蔽正样本，只保留负样本相似度
        sim_matrix_A = sim_matrix_A * neg_mask - 1e9 * (1 - neg_mask)
        # 选取top-k难负样本
        hard_k = max(1, int(self.hard_ratio * z_A.shape[0]))
        hard_neg_vals_A, hard_neg_idx_A = torch.topk(sim_matrix_A, k=hard_k, dim=1)
        # 构建难负样本掩码
        hard_neg_mask_A = torch.zeros_like(sim_matrix_A).scatter_(1, hard_neg_idx_A, 1.0)

        # B模态内同理
        sim_matrix_B = torch.matmul(z_B, z_B.T) / self.temperature
        sim_matrix_B = sim_matrix_B * neg_mask - 1e9 * (1 - neg_mask)
        hard_neg_vals_B, hard_neg_idx_B = torch.topk(sim_matrix_B, k=hard_k, dim=1)
        hard_neg_mask_B = torch.zeros_like(sim_matrix_B).scatter_(1, hard_neg_idx_B, 1.0)

        # 3. 双向跨模态对比（可选）：药物-靶点跨模态负样本
        cross_neg_loss = 0.0
        if cross_index is not None:
            cross_neg_mask = (cross_index != cross_index.T).float()
            cross_sim = torch.matmul(z_A, z_B.T) / self.temperature
            cross_neg = (torch.exp(cross_sim) * cross_neg_mask).sum(dim=1) + 1e-8
            cross_neg_loss = -torch.log(pos_sim_exp / (pos_sim_exp + cross_neg)).mean() * 0.1

        # 4. 难负样本损失计算
        neg_denom_A = (torch.exp(sim_matrix_A) * hard_neg_mask_A).sum(dim=1) + 1e-8
        loss_A = -torch.log(pos_sim_exp / (pos_sim_exp + neg_denom_A))

        neg_denom_B = (torch.exp(sim_matrix_B) * hard_neg_mask_B).sum(dim=1) + 1e-8
        loss_B = -torch.log(pos_sim_exp / (pos_sim_exp + neg_denom_B))

        # 融合跨模态对比损失
        total_loss = 0.5 * (loss_A.mean() + loss_B.mean()) + cross_neg_loss
        return total_loss

class GatedSoftAttentionFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # 注意力得分网络
        self.attention_score = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features):
        # 计算基础注意力得分
        score = self.attention_score(features)  # (B, N, 1)
        # 计算门控系数
        gate = self.gate(features)  # (B, N, 1)
        # 门控调节后的注意力权重
        weights = score * gate
        weights = torch.softmax(weights, dim=1)  # (B, N, 1)

        # 特征融合
        fused = torch.sum(features * weights, dim=1)  # (B, D)
        return fused, weights




class GIN_Model(nn.Module):
    def __init__(self, in_feats, hidden_feats=128, num_heads=4):
        super().__init__()
        self.gnn = GraphormerGIN(
            dim_h=hidden_feats,
            num_node_features=in_feats,
            num_heads=num_heads
        )

    def forward(self, batch_graph):
        # 训练时增强图
        if self.training:
            batch_graph = augment_graph(batch_graph)

        node_feats = batch_graph.ndata['h']
        node_embeds = self.gnn(batch_graph, node_feats)
        graph_embed = node_embeds.mean(dim=0)

        return graph_embed



class ProteinCNN(nn.Module):
    def __init__(self, embedding_dim, conv_dim):
        super(ProteinCNN, self).__init__()

        self.embedding = nn.Embedding(26, embedding_dim)
        # 1-gram, 3-gram, 5-gram)
        self.conv1_1 = nn.Conv1d(in_channels=embedding_dim, out_channels=conv_dim, kernel_size=1)
        self.conv1_2 = nn.Conv1d(in_channels=conv_dim, out_channels=conv_dim * 2, kernel_size=1)
        self.conv1_3 = nn.Conv1d(in_channels=conv_dim * 2, out_channels=conv_dim, kernel_size=1)

        self.conv3_1 = nn.Conv1d(in_channels=embedding_dim, out_channels=conv_dim, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv1d(in_channels=conv_dim, out_channels=conv_dim * 2, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv1d(in_channels=conv_dim * 2, out_channels=conv_dim, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv1d(in_channels=embedding_dim, out_channels=conv_dim, kernel_size=5, padding=2)
        self.conv5_2 = nn.Conv1d(in_channels=conv_dim, out_channels=conv_dim * 2, kernel_size=5, padding=2)
        self.conv5_3 = nn.Conv1d(in_channels=conv_dim * 2, out_channels=conv_dim, kernel_size=5, padding=2)

        self.gelu = nn.GELU()
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, embedding_dim) -> (batch, embedding_dim, seq_len)

        x1 = self.gelu(self.conv1_1(x))  # (batch, 128, seq_len) -> (batch, 512, 1)
        x1 = self.gelu(self.conv1_2(x1))
        x1 = self.pool(self.conv1_3(x1)).squeeze(-1)

        x3 = self.gelu(self.conv3_1(x))  # (batch, 128, seq_len) -> (batch, 512, 1)
        x3 = self.gelu(self.conv3_2(x3))
        x3 = self.pool(self.conv3_3(x3)).squeeze(-1)

        x5 = self.gelu(self.conv5_1(x))  # (batch, 128, seq_len) -> (batch, 512, 1)
        x5 = self.gelu(self.conv5_2(x5))
        x5 = self.pool(self.conv5_3(x5)).squeeze(-1)

        x = x1 + x3 + x5
        return x



class KANLayer1(nn.Module):
    """可学习的非线性函数层（简化版 KAN 层）"""

    def __init__(self, input_dim, output_dim, num_basis=8):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)  # 线性部分
        self.basis_functions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, num_basis),  # 样条基函数
                nn.Linear(num_basis, 1, bias=False)
            ) for _ in range(input_dim)
        ])

    def forward(self, x):
        # 线性变换
        linear_out = self.linear(x)

        # 非线性变换（对每个输入维度独立处理）
        nonlinear_out = torch.zeros_like(linear_out)
        for i in range(x.shape[1]):
            basis_input = x[:, i:i + 1]  # 取第 i 个特征
            nonlinear_out[:, i] = self.basis_functions[i](basis_input).squeeze()

        return linear_out + nonlinear_out  # 线性 + 非线性

class TriDTI(nn.Module):
    def __init__(
            self,
            hidden_dim=512,
            projection_dim=128,
            mol_dim=768,
            prot_dim=320,
            gcn_dim=128,
            cnn_dim=128,
            drug_atom_dim=79,
            drug_graph_dim=769,
            prot_graph_dim=1281,
            num_heads=8,
    ):
        super().__init__()

        self.contrastive_loss = HardNegativeContrastiveLoss(temperature=0.1, hard_ratio=0.4)

        self.gcn1 = GIN_Model(in_feats=drug_atom_dim, hidden_feats=gcn_dim)
        self.protein1 = ProteinCNN(embedding_dim=cnn_dim, conv_dim=cnn_dim)
        # self.gcn2 = GATv2_Model(in_feats=drug_graph_dim, hidden_feats=gcn_dim)
        # self.gcn3 = GATv2_Model(in_feats=prot_graph_dim, hidden_feats=gcn_dim)
        # self.gcn2 = GraphSAGE_Model(in_feats=drug_graph_dim, hidden_feats=gcn_dim)
        # self.gcn3 = GraphSAGE_Model(in_feats=prot_graph_dim, hidden_feats=gcn_dim)
        # 在TriDTI模型的__init__中
        self.gcn2 = AdaptiveGAT(
            in_feats=drug_graph_dim,  # 你的药物图输入维度（如64/128）
            hidden_feats=128,  # 强制128维输出
            n_layers=2,
            n_heads=4,
            dropout=0.1
        )
        self.gcn3 = AdaptiveGAT(
            in_feats=prot_graph_dim,
            hidden_feats=128,
            n_layers=2,
            n_heads=4,
            dropout=0.1
        )

        self.proj_mol_llm = nn.Sequential(
            nn.Linear(mol_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, projection_dim)
        )

        self.proj_mol_graph = nn.Sequential(
            nn.Linear(gcn_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, projection_dim)
        )
        # self.proj_mol_ddi = nn.Sequential(
        #     nn.Linear(gcn_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(hidden_dim, projection_dim)
        # )
        # 在TriDTI模型中，proj_mol_ddi的定义应如下（匹配AdaptiveGAT输出）
        self.proj_mol_ddi = nn.Sequential(
            nn.Linear(128, 128),  # 输入128，输出128（可根据需求调整）
            nn.ReLU(),
            nn.Dropout(0.1)
        )#davis

        # self.proj_mol_ddi = nn.Sequential(
        #     nn.Linear(128, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(256, 64)
        # )#biosnap,drugbank

        self.proj_prot_llm = nn.Sequential(
            nn.Linear(prot_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, projection_dim)
        )

        self.proj_prot_cnn = nn.Sequential(
            nn.Linear(cnn_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, projection_dim)
        )
        self.proj_prot_ppi = nn.Sequential(
            nn.Linear(gcn_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, projection_dim)
        )#davis

        # self.proj_prot_ppi = nn.Sequential(
        #     nn.Linear(128, 128),  # 输入维度改为128，匹配实际输入
        #     nn.ReLU(),
        #     nn.Linear(128, 64)
        # )#biosnap

        # self.drug_fusion_module = SoftAttentionFusion(input_dim=projection_dim, hidden_dim=projection_dim)
        # self.prot_fusion_module = SoftAttentionFusion(input_dim=projection_dim, hidden_dim=projection_dim)

        self.drug_fusion_module = GatedSoftAttentionFusion(
            input_dim=projection_dim, hidden_dim=projection_dim
        )
        self.prot_fusion_module = GatedSoftAttentionFusion(
            input_dim=projection_dim, hidden_dim=projection_dim
        )

        self.cross_attention_drug = nn.MultiheadAttention(embed_dim=projection_dim, num_heads=num_heads,
                                                          batch_first=True)
        self.cross_attention_prot = nn.MultiheadAttention(embed_dim=projection_dim, num_heads=num_heads,
                                                          batch_first=True)

        self.cross_attention = DoubleCrossAttention(embed_dim=projection_dim, num_heads=num_heads)

        self.mlp = nn.Sequential(
            KANLayer1(projection_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.initialize_weights()

    def initialize_weights(self):
        """ Xavier Initialization """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, mg_list, dg_list, pg_list, drug_embedding, prot_embedding, d_id, p_id, dg_index, pg_index,
                protein_sequence):

        drug_embedding = drug_embedding.float()  # 药物嵌入→float32（线性层用）
        prot_embedding = prot_embedding.float()  # 蛋白嵌入→float32（线性层用）
        # protein_sequence保持long类型（Embedding层需要整型索引）
        protein_sequence = protein_sequence.long() if protein_sequence.dtype != torch.long else protein_sequence

        drug_llm = self.proj_mol_llm(drug_embedding)
        drug_graph = self.proj_mol_graph(torch.stack([self.gcn1(g) for g in mg_list]))
        # ddi_graph = self.proj_mol_ddi(torch.stack([self.gcn2(g, idx) for g, idx in zip(dg_list, dg_index)]))

        prot_llm = self.proj_prot_llm(prot_embedding)
        prot_cnn = self.proj_prot_cnn(self.protein1(protein_sequence))
        # ppi_graph = self.proj_prot_ppi(torch.stack([self.gcn3(g, idx) for g, idx in zip(pg_list, pg_index)]))

        # drug_llm = torch.zeros_like(drug_llm)
        # prot_llm = torch.zeros_like(prot_llm)

        # 正确代码（传图g + 节点特征g.ndata['h'] + 索引idx）
        ddi_graph = self.proj_mol_ddi(
            torch.stack([self.gcn2(g, g.ndata['h'], idx) for g, idx in zip(dg_list, dg_index)]))
        ppi_graph = self.proj_prot_ppi(
            torch.stack([self.gcn3(g, g.ndata['h'], idx) for g, idx in zip(pg_list, pg_index)]))

        kd_loss_drug_mol = self.contrastive_loss(drug_llm, drug_graph, d_id, cross_index=p_id)
        kd_loss_drug_ddi = self.contrastive_loss(drug_llm, ddi_graph, d_id)

        kd_loss_prot_cnn = self.contrastive_loss(prot_llm, prot_cnn, p_id)
        kd_loss_prot_ppi = self.contrastive_loss(prot_llm, ppi_graph, p_id)

        total_kd_loss = kd_loss_drug_mol + kd_loss_drug_ddi + kd_loss_prot_cnn + kd_loss_prot_ppi

        drug_features_for_fusion = torch.stack([drug_llm, drug_graph, ddi_graph], dim=1)  # (B, 3, projection_dim)
        prot_features_for_fusion = torch.stack([prot_llm, prot_cnn, ppi_graph], dim=1)  # (B, 3, projection_dim)

        fused_drug_feature, drug_fusion_weights = self.drug_fusion_module(
            drug_features_for_fusion)  # (B, projection_dim)
        fused_prot_feature, prot_fusion_weights = self.prot_fusion_module(
            prot_features_for_fusion)  # (B, projection_dim)



        # drug_cross_attn, _ = self.cross_attention_drug(fused_drug_feature.unsqueeze(1), fused_prot_feature.unsqueeze(1),
        #                                                fused_prot_feature.unsqueeze(1))
        # prot_cross_attn, _ = self.cross_attention_prot(fused_prot_feature.unsqueeze(1), fused_drug_feature.unsqueeze(1),
        #                                                fused_drug_feature.unsqueeze(1))

        # drug_final_feature = (drug_cross_attn.squeeze(1) + fused_drug_feature)
        # prot_final_feature = (prot_cross_attn.squeeze(1) + fused_prot_feature)

        drug_cross, prot_cross = self.cross_attention(fused_drug_feature.unsqueeze(1), fused_prot_feature.unsqueeze(1))
        drug_final_feature = drug_cross.squeeze(1) + fused_drug_feature
        prot_final_feature = prot_cross.squeeze(1) + fused_prot_feature

        x = torch.cat([drug_final_feature, prot_final_feature], dim=-1)
        cls_out = self.mlp(x).squeeze(-1)



        return cls_out, total_kd_loss  # drug_fusion_weights, prot_fusion_weights