from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, station_path, nwp_path, mode,input_steps, pred_steps,
                 patch_size=1, train_test_split=0.8, normalization="min-max"):
        """
            @params: input_steps 控制输入的时间步数
                     pred_steps 当pred_steps=1时,为回归任务,多时间步回归或单时间步回归由input_steps控制
                                当pred_steps>1时,为预测任务
        """
        self.mode = mode
        self.train_test_split = train_test_split
        self.normalization = normalization
        self.input_steps = input_steps
        self.pred_steps = pred_steps
        self.max_power_list = np.array(pd.read_csv('/home/yztang/encoder_with_transformer/station_info_118.csv'))[:, -1]
        
        # 加载3*3的nwp数据
        all_nwp = []
        nwps = sorted(os.listdir(nwp_path))
        for nwp in nwps:
            nwp_data = np.load(os.path.join(nwp_path, nwp))
            all_nwp.append(nwp_data)
            
        self.all_nwp = np.stack(all_nwp).transpose(1, 0, 2, 3, 4)   # [34954, 118, 7, 3, 3]
        
        all_stations = []
        all_station_power = []
        
        stations = sorted(os.listdir(station_path))
        
        for station in stations:
            station_data = np.array(pd.read_csv(os.path.join(station_path, station)))
            
            station_power = station_data[:, -1]
            station_data = station_data[:, 0:10]
            
            all_stations.append(station_data)
            all_station_power.append(station_power)
        
        self.all_stations = np.stack(all_stations).transpose(1, 0, 2)    # 场站数据[34954, 118, 10]
        self.all_station_power = np.stack(all_station_power).transpose(1, 0)   # 场站功率[34954, 118]
        
        # 分别计算不同来源的数据的归一化参数
        station_min, station_max = self.compute_min_max(self.all_stations)
        nwp_min, nwp_max = self.compute_min_max(self.all_nwp)
        
        # 归一化
        self.all_stations = self.normalize(self.all_stations, station_min, station_max)
        # 网格数据归一化
        self.all_nwp = self.normalize(self.all_nwp, nwp_min, nwp_max)
        # 功率除以装机容量，对功率做归一化
        for i in range(118):
            self.all_station_power[ :, i] = self.all_station_power[:, i] / self.max_power_list[i]
        
        self.train_end = int (self.all_stations.shape[0] * train_test_split)
        
        if self.mode == 'train':
            self.all_nwp = self.all_nwp[:self.train_end, :, :, :, :]
            self.all_stations = self.all_stations[:self.train_end, :, :]
            self.all_station_power = self.all_station_power[ :self.train_end, ]
            print("输入数据长度:", self.all_nwp.shape[0])
            print("按input_steps划分后的数据长度:", self.__len__())
        elif self.mode == 'val':
            self.all_nwp = self.all_nwp[self.train_end:, :, :, :, :]
            self.all_stations = self.all_stations[self.train_end: , :, :]
            self.all_station_power = self.all_station_power[self.train_end: , ]
    def compute_min_max(self, data_source):
        # 如果数据来源是路径，则遍历文件获取数据，否则直接计算
        if isinstance(data_source, str):  # 数据来源是路径
            all_data = []
            stations = sorted(os.listdir(data_source))
            for station in stations:
                # 单站:[34954, 11]
                station_data = np.array(pd.read_csv(os.path.join(data_source, station)))[:, :10]
                all_data.append(station_data)
            all_data = np.stack(all_data, axis=0).transpose(1, 0, 2) # [34954, 118, 9]
        else:  # 数据来源是npy文件
            all_data = data_source
        
        return np.min(all_data, axis=0), np.max(all_data, axis=0)

    def __len__(self):
        return int((self.all_nwp.shape[0] - self.input_steps - self.pred_steps + 1) /24)

    def __getitem__(self, index):
        index = index * 24
        input_station = self.all_stations[index:index + self.input_steps, :, :]
        input_nwp = self.all_nwp[index:index + self.input_steps, :, :, :, :]
        
        if self.pred_steps == 1:
            target_power = self.all_station_power[index + self.input_steps, :]
        else:
            target_power = self.all_station_power[index + self.input_steps:index + self.input_steps + self.pred_steps, :]
        
        return input_station, target_power, input_nwp

    def normalize(self, data, data_min, data_max):
        if self.normalization == "min-max":
            return (data - data_min) / (data_max - data_min)
        elif self.normalization == "z-score":
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            return (data - mean) / std

    def denormalize(self, data, data_min, data_max):
        if self.normalization == "min-max":
            return data * (data_max - data_min) + data_min
        elif self.normalization == "z-score":
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            return data * std + mean

if __name__ == '__main__':
    data = CustomDataset(station_path='/home/yztang/encoder_with_transformer/processed_data_118', 
                         nwp_path='/home/yztang/encoder_with_transformer/station_nwp_118', 
                         mode='train', input_steps=12, pred_steps=12)
    print(data.__len__())
    dataloader = DataLoader(data, batch_size=2, shuffle=False)
    num = 0
    for station_data, station_power, nwp in dataloader:
        num += 1
        # 返回的形状:station_nwp:[bs, 118, 10] station_power:[bs, 118] nwp:[bs, 7, 41, 41] 
        print(station_data.shape, station_power.shape, nwp.shape)
        print(num)