from pathlib import Path
import open3d as o3d
import torch
import numpy as np
import torch.utils.data as data
import h5py
import transforms3d


class PCN_pcd(data.Dataset):
    def __init__(self, path, prefix="train"):
        if prefix in ["train", "val", "test"]:
            self.prefix = prefix
            self.file_path = Path(path).joinpath(prefix)
        else:
            raise ValueError("ValueError prefix should be [train/val/test] ")
        
        # bool for transform 
        self.scale = 0
        self.mirror = 1
        self.rot = 0
        # bool for sample 
        self.sample = 1
        # label map
        self.label_map ={'02691156': '0', '02933112': '1', '02958343': '2',
                         '03001627': '3', '03636649': '4', '04256520': '5',
                         '04379243': '6', '04530566': '7', 'all': '8'}
        # 获取 partial 路径、gt 路径、labels 标签
        self.partial_path, self.gt_path, self.labels = self.get_data(self.file_path)


    def __len__(self):
        return len(self.partial_path)


    def read_pcd(self, path):
        pcd = o3d.io.read_point_cloud(str(path))
        points = np.asarray(pcd.points)
        return points


    def get_data(self, file_path): 
        partial_parent = Path(file_path).joinpath('partial')
        gt_parent = Path(file_path).joinpath('complete')
        partial_data, gt_data, labels = [], [], []
        
        # 同时构建 partial 和对应的 gt 路径
        for c in partial_parent.iterdir():
            category = c.name  # 类别名（如 02691156）
            for obj in c.iterdir():
                obj_name = obj.name  # 对象名（如 obj001）
                
                # 添加 partial 路径列表
                obj_list = [str(f) for f in obj.iterdir()]
                partial_data.append(obj_list)
                
                # 基于 partial 构建对应的 gt 路径
                gt_path = gt_parent / category / f"{obj_name}.pcd"
                gt_data.append(str(gt_path))
                
                # 添加标签
                labels.append(self.label_map[category])
        
        return partial_data, gt_data, labels


    def upsample(self, ptcloud, n_points):
        curr = ptcloud.shape[0]
        need = n_points - curr
        if need < 0:
            choice = np.random.permutation(ptcloud.shape[0])
            return ptcloud[choice[:n_points]]

        while curr <= need:
            ptcloud = np.tile(ptcloud, (2, 1))
            need -= curr
            curr *= 2

        choice = np.random.permutation(ptcloud.shape[0])
        ptcloud = np.concatenate((ptcloud, ptcloud[choice[:need]]))

        return ptcloud


    def get_transform(self, points):
        result = []
        rnd_value = np.random.uniform(0, 1)

        if self.mirror and self.prefix == 'train':
            trfm_mat = transforms3d.zooms.zfdir2mat(1)
            trfm_mat_x = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), trfm_mat)
            trfm_mat_z = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), trfm_mat)
            if rnd_value <= 0.25:
                trfm_mat = np.dot(trfm_mat_x, trfm_mat)
                trfm_mat = np.dot(trfm_mat_z, trfm_mat)
            elif rnd_value <= 0.5:
                trfm_mat = np.dot(trfm_mat_x, trfm_mat)
            elif rnd_value <= 0.75:
                trfm_mat = np.dot(trfm_mat_z, trfm_mat)

        for ptcloud in points:
            ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)

            if self.scale:
                ptcloud = ptcloud * self.scale
            result.append(ptcloud)

        return result[0],result[1]


    def __getitem__(self, index):
        # 获取 partial 路径列表
        partial_path_list = self.partial_path[index]
        n_sample = len(partial_path_list)
        idx = np.random.randint(0, n_sample)
        partial_path = partial_path_list[idx]

        # 读取并处理 partial 点云
        partial = self.read_pcd(partial_path)
        partial = self.upsample(partial, 2048)

        # 读取 complete 点云
        gt_path = self.gt_path[index]
        complete = self.read_pcd(gt_path)

        # 应用变换
        if self.prefix == 'train':
            partial, complete = self.get_transform([partial, complete])
        
        # 转换为 PyTorch 张量
        complete = torch.from_numpy(complete)
        partial = torch.from_numpy(partial)
        
        # 使用预加载的标签
        label = self.labels[index]
        
        # 从路径提取对象名（用于测试）
        obj = Path(partial_path).parent.name
        
        if self.prefix == 'test':
            return label, partial, complete, obj
        else:
            return label, partial, complete


class C3D_h5(data.Dataset):
    def __init__(self, path, prefix="train"):
        if prefix in ["train", "val", "test"]:
            self.prefix = prefix
            self.file_path = Path(path).joinpath(prefix)
        else:
            raise ValueError("ValueError prefix should be [train/val/test] ")
        # bool for transform
        self.scale = 1
        self.mirror = 1
        self.rot = 0
        # bool for sample
        self.sample = 1
        # label map
        self.label_map ={'02691156': '0', '02933112': '1', '02958343': '2',
                         '03001627': '3', '03636649': '4', '04256520': '5',
                         '04379243': '6', '04530566': '7', 'all': '8'}
        # 获取 partial 路径、gt 路径、labels 标签
        self.partial_path, self.gt_path, self.labels = self.get_data(self.file_path)


    def __len__(self):
        return len(self.partial_path)


    def get_data(self, file_path):
        partial_parent = Path(file_path).joinpath('partial')
        gt_parent = Path(file_path).joinpath('gt')
        partial_data, gt_data, labels = [], [], []
        for c in partial_parent.iterdir():
            category = c.name  # 类别名（如 02691156）
            for obj in c.iterdir():
                obj_name = obj.name  # 对象名（如 obj001.h5）
                # 添加 partial 路径列表
                partial_data.append(str(obj))
                # 添加 gt 路径列表
                gt_data.append(str(gt_parent.joinpath(category, obj_name)))
                if self.prefix == "test":
                    labels.append(obj_name)
                else:
                    labels.append(self.label_map[category])
                    
        return partial_data, gt_data, labels


    def upsample(self, ptcloud, n_points):
        curr = ptcloud.shape[0]
        need = n_points - curr
        if need < 0:
            choice = np.random.permutation(ptcloud.shape[0])
            return ptcloud[choice[:n_points]]
        while curr <= need:
            ptcloud = np.tile(ptcloud, (2, 1)) # 复制
            need -= curr
            curr *= 2
        choice = np.random.permutation(ptcloud.shape[0])
        ptcloud = np.concatenate((ptcloud, ptcloud[choice[:need]]))
        return ptcloud


    def get_transform(self, points):
        result = []
        rnd_value = np.random.uniform(0, 1)
        angle = np.random.uniform(0,2*np.pi)
        scale = np.random.uniform(1/1.6, 1)

        trfm_mat = transforms3d.zooms.zfdir2mat(1)
        if self.mirror and self.prefix == 'train':

            trfm_mat_x = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), trfm_mat)
            trfm_mat_z = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), trfm_mat)
            if rnd_value <= 0.25:
                trfm_mat = np.dot(trfm_mat_x, trfm_mat)
                trfm_mat = np.dot(trfm_mat_z, trfm_mat)
            elif rnd_value <= 0.5:
                trfm_mat = np.dot(trfm_mat_x, trfm_mat)
            elif rnd_value <= 0.75:
                trfm_mat = np.dot(trfm_mat_z, trfm_mat)
        if self.rot:
                trfm_mat = np.dot(transforms3d.axangles.axangle2mat([0,1,0],angle), trfm_mat)

        for ptcloud in points:
            ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)
            if self.scale:
                ptcloud = ptcloud * scale
            result.append(ptcloud)

        return result[0],result[1]


    def __getitem__(self, index):
        partial_path = self.partial_path[index]
        with h5py.File(partial_path, 'r') as f:
            partial = np.array(f['data'])

        if self.prefix == 'train' and self.sample:
            partial = self.upsample(partial, 2048) # 将点云采样/扩充到2048个点

        # 这里是因为C3D数据集的test集没有gt
        if self.prefix not in ["test"]:
            complete_path = self.gt_path[index]
            with h5py.File(complete_path, 'r') as f:
                complete = np.array(f['data'])
            if self.prefix == 'train':
                partial, complete = self.get_transform([partial, complete])

            # 转换为 PyTorch 张量
            complete = torch.from_numpy(complete)
            partial = torch.from_numpy(partial)

            # 使用预加载的标签
            label = self.labels[index]

            return label, partial, complete
        else:
            partial = torch.from_numpy(partial)
            label = self.labels[index]
            return label, partial, partial


if __name__ == '__main__':
    dataset = C3D_h5(prefix='test')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=True, num_workers=0)
    for idx, data in enumerate(dataloader, 0):
        print(data.shape)




