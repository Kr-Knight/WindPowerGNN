import torch
import pandas as pd
import numpy as np

def get_static_adj(geo_data, sigma=100.0, theta=10.0, device='cuda'):
    """
    生成基于地理距离的高斯核邻接矩阵
    :param geo_data:  [118, 2] 经度(longitude)和纬度(latitude)
    :param sigma:     高斯核宽度参数（控制衰减速度）
    :param theta:     距离阈值（公里），超过则置零
    :param device:    计算设备
    :return:          [118, 118]邻接矩阵
    """
    # 转换为弧度制
    geo_rad = torch.deg2rad(geo_data).to(device)  # [118, 2]
    
    # 拆分经纬度
    lon = geo_rad[:, 0].unsqueeze(1)  # [118, 1]
    lat = geo_rad[:, 1].unsqueeze(1)  # [118, 1]

    # 计算Haversine距离矩阵
    dlon = lon - lon.T  # [118, 118]
    dlat = lat - lat.T  # [118, 118]
    
    a = torch.sin(dlat/2.0)**2 + torch.cos(lat) * torch.cos(lat.T) * torch.sin(dlon/2.0)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
    dist_matrix = 6371.0 * c  # 地球半径6371km，得到公里级距离矩阵

    # 高斯核转换
    adj_matrix = torch.exp(-dist_matrix**2 / (2 * sigma**2))  # [118, 118]
    
    # 应用距离阈值（可选）
    adj_matrix[dist_matrix > theta] = 0
    
    # 对角线置零（不包含自连接）
    adj_matrix.fill_diagonal_(0)
    
    # 对称化处理
    adj_matrix = (adj_matrix + adj_matrix.T) / 2  # 确保对称
    
    return adj_matrix


def haversine_vectorized(lon1, lat1, lon2, lat2):
    """
    向量化Haversine距离计算
    输入：
        lon1: [n] 经度（度）
        lat1: [n] 纬度（度）
        lon2: [m] 经度（度）
        lat2: [m] 纬度（度）
    返回：
        dist_matrix: [n, m] 距离矩阵（公里）
    """
    # 转换为弧度
    lon1, lat1, lon2, lat2 = map(torch.deg2rad, [lon1, lat1, lon2, lat2])
    
    # 扩展维度用于广播计算
    dlon = lon2.unsqueeze(0) - lon1.unsqueeze(1)
    dlat = lat2.unsqueeze(0) - lat1.unsqueeze(1)
    a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
    c = 2 * torch.asin(torch.sqrt(a))
    km = 6371 * c
    return km 

def build_static_edges(coords, method='topk', threshold_km=10, k=5):
    """
    构建静态图边关系
    参数：
        coords: [num_nodes, 2] 经纬度坐标
        method: 'threshold' 或 'topk'
        threshold_km: 距离阈值（公里）
        k: 最近邻数量
    返回：
        edge_index: [2, num_edges] 的np.array
    """
    if torch.cuda.is_available():
        coords = coords.cuda()
    
    # 计算距离矩阵
    lon = coords[:, 0]
    lat = coords[:, 1]
    dist_mat = haversine_vectorized(lon, lat, lon, lat)
    
    # 构建邻接矩阵
    if method == 'threshold':
        adj = (dist_mat <= threshold_km) & (dist_mat > 0)  # 排除自环
    elif method == 'topk':
        adj = torch.zeros_like(dist_mat, dtype=torch.bool)
        sorted_idx = torch.argsort(dist_mat, dim=1)
        for i in range(coords.size(0)):
            adj[i, sorted_idx[i, 1:k+1]] = 1  # 取前k个（排除自身）
    
    # 转换为边索引
    nonzero_indices = torch.nonzero(adj, as_tuple=False)
    rows = nonzero_indices[:, 0]
    cols = nonzero_indices[:, 1]
    return torch.stack([rows, cols], dim=0)  # [2, num_edges]


# 使用示例
if __name__ == '__main__':
    # 模拟数据：118个风电场的经纬度（假设数据范围：经度-180~180，纬度-90~90）
    geo_data = torch.Tensor(np.array(pd.read_csv('/home/yztang/encoder_with_transformer/station_info_118.csv'))[:, 1:3])
    #geo_data = torch.randn(118, 2) * 30 + torch.tensor([116.4, 39.9])  # 北京周边随机分布
    
    # adj = get_static_adj(geo_data, sigma=50.0, theta=200.0)
    # print("Adjacency Matrix Shape:", adj.shape)
    # print("Sample Weights:\n", adj[:3, :3])
    static_edge = build_static_edges(geo_data)
    print(static_edge.shape)