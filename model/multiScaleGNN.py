import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from multiScaleGNN_module import ConvSC, sampling_generator

class WindFarmModel(nn.Module):
    def __init__(self, node_feat_dim=26, hidden_dim=64):
        super().__init__()
        
        # 局部气象特征提取 (3x3区域)
        self.local_feat_extract = Encoder(C_in=7, C_hid=8, N_S=2, spatio_kernel=3)
        # 场站特征映射
        self.station_mlp = nn.Linear(10, 32)
        
        # 空间编码器 (GATv2 with dynamic edges)
        self.gat1 = GATv2Conv(node_feat_dim, hidden_dim, edge_dim=1)
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=1)
        
        # 时间编码器 (TCN + Transformer)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2, padding=2),
        )
        self.time_attn = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4)
        
        # 融合层
        self.fusion = nn.Linear(hidden_dim+1, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, 1)
        
        # 地理相关参数
        self.dist_matrix = None  # 预计算的地理距离矩阵

    def forward(self, x_base, x_local, edge_index):
        """
        x_base: [batch, time_steps, num_nodes, 10]
        x_local: [batch, time_steps, num_nodes, 7, 3, 3]
        edge_index: [2, num_edges] 静态边
        """
        batch_size, time_steps, num_nodes = x_base.shape[:3]
        
        # 1. 局部特征提取
        x_local = x_local.view(-1, 7, 3, 3)
        local_feat = self.local_feat_extract(x_local)  # [batch, time, 8, nodes]
        local_feat = local_feat.view(batch_size, num_nodes, time_steps, -1)  # [batch, nodes, time, 8]
        
        # 2. 特征拼接
        x_base = x_base.permute(0,2,1,3)  # [batch, nodes, time, 10]
        x_base = self.station_mlp(x_base)
        node_feat = torch.cat([x_base, local_feat], dim=-1)  # [batch, nodes, time, 18]
        
        # 3. 时空处理
        all_outputs = []
        for t in range(time_steps):
            # 动态边计算 (特征相似性)
            dyn_edge_index = self.compute_dynamic_edges(node_feat[:,:,t,:]) 
            
            # 合并批量维度和节点维度
            x = node_feat[:,:,t,:].reshape(-1, node_feat.size(-1))
            batch_edge_index = self.adjust_edge_index(edge_index, dyn_edge_index, batch_size, num_nodes)
            
            # 空间聚合 
            h = self.gat1(x, batch_edge_index)
            h = F.relu(h)
            h = self.gat2(h, batch_edge_index)
            
            # 恢复批量维度
            h = h.reshape(batch_size, num_nodes, -1)
            
            # 时间处理
            if t > 0:
                h_time = self.temporal_conv(h.permute(0, 2, 1)).permute(0, 2, 1)
                h = h + h_time  # 残差连接
                
            all_outputs.append(h)
        
        # 4. 时间注意力
        temporal_out = torch.stack(all_outputs, dim=1)  # [batch, time, nodes, hid]
        temporal_out = temporal_out.reshape(-1, num_nodes, temporal_out.size(-1))
        temporal_out = self.time_attn(temporal_out)
        temporal_out = temporal_out.reshape(batch_size, -1, num_nodes, temporal_out.size(-1))
        
        # 5. 多尺度融合
        all_preds = []
        for t in range(time_steps):
            spatial_out = torch.mean(temporal_out[:, t, :, :], dim=-1, keepdim=True)  # 空间维度聚合
            temporal_out_t = torch.max(temporal_out[:, :, :, :], dim=1)[0]  # 时间维度聚合
            
            fused = self.fusion(torch.cat([spatial_out, temporal_out_t], dim=-1))
            pred = self.regressor(fused).squeeze(-1)  # [batch, nodes]
            all_preds.append(pred)
        
        all_preds = torch.stack(all_preds, dim=1)  # [batch, time, nodes]
        
        return all_preds

    def compute_dynamic_edges(self, node_feat, k=5):
        """
        param: node_feat:[batch, nodes, 18]
        动态计算Top-K特征相似性边
        返回edge_index: [2, num_edges]
        """
        batch_size, num_nodes = node_feat.shape[:2]
        cos_sim = F.cosine_similarity(node_feat.unsqueeze(1), node_feat.unsqueeze(2), dim=-1)
        topk = torch.topk(cos_sim, k=k, dim=-1)
        edge_index = []
        device = node_feat.device  # 获取node_feat所在的设备
        for b in range(batch_size):
            src_nodes = torch.arange(num_nodes).unsqueeze(-1).expand(-1, k).flatten().to(device)
            dst_nodes = topk.indices[b].flatten()
            batch_edge_index = torch.stack([src_nodes, dst_nodes], dim=0)
            edge_index.append(batch_edge_index)
        edge_index = torch.cat(edge_index, dim=1)
        return edge_index

    def adjust_edge_index(self, edge_index, dyn_edge_index, batch_size, num_nodes):
        batch_edge_index = []
        for b in range(batch_size):
            batch_edge_index.append(edge_index + b * num_nodes)
        batch_edge_index = torch.cat(batch_edge_index, dim=1)
        return torch.cat([batch_edge_index, dyn_edge_index], dim=1)
    # def geo_correlation_loss(self, pred, targets):
    #     """地理相关性惩罚项"""
    #     if self.dist_matrix is None:
    #         # 预计算距离矩阵 (Haversine公式)
    #         self.dist_matrix = compute_geo_distance(loc_coords)
            
    #     pairwise_diff = pred.unsqueeze(1) - pred.unsqueeze(2)  # [batch, n, n]
    #     dist_penalty = torch.exp(-self.dist_matrix / 50.0)  # 距离衰减系数
    #     loss = torch.mean(dist_penalty * pairwise_diff.pow(2))
    #     return loss
    
    
class Encoder(nn.Module):
    def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True):
        super(Encoder, self).__init__()
        samplings = sampling_generator(N_S)
        self.enc = nn.Sequential(
              ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0],
                     act_inplace=act_inplace),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s,
                     act_inplace=act_inplace) for s in samplings[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent
    
if __name__ == '__main__':
    model = WindFarmModel()

    x_base = torch.randn(16, 12, 118, 10)
    x_local = torch.randn(16, 12, 118, 7, 3, 3)
    edge_index = torch.randint(0, 118, (2, 1234))

    pred = model(x_base, x_local, edge_index)
    print(pred.shape)
    # node_feat = torch.randn((15, 118, 26))
    # print(model.compute_dynamic_edges(node_feat).shape)