from torch.utils.data import Dataset
import torch
import numpy as np
import h5py

class DatasetH5(Dataset):
    def __init__(
        self,
        filepath,
        image_size,
        dynamic_sampling = False,
        ext = 'h5',
    ):
        super().__init__()
        self.filepath = filepath
        self.image_size = image_size
        self.dynamic_sampling = dynamic_sampling
        
        if dynamic_sampling:
            self.col_name    = "train/mean"
            self.colstd_name = "train/std"
        else:
            self.col_name = "train/latent"

        col_min = "train/min"
        col_max = "train/max"
        
        print(f"Checking dataset from column {self.col_name}")
        self.h5_handler = h5py.File(self.filepath, 'r')

        len_data = len(self.h5_handler.get(self.col_name))
        data_min = np.asarray(self.h5_handler.get(col_min))
        data_max = np.asarray(self.h5_handler.get(col_max))

        self.global_min = np.min(data_min)
        self.global_max = np.max(data_max)
        self.len_data = len_data
        print(f"{self.len_data = }")
        print(f"{self.global_min = } {self.global_max = }")

    def __len__(self):
        return self.len_data

    def sample(self, mu, std):
        # we do this in pytorch to keep it consistent with the original code
        x = torch.from_numpy(mu) + torch.from_numpy(std) * torch.randn(mu.shape)
        return x.numpy()

    def __getitem__(self, index):
        # print(f"{index = }")
        if self.dynamic_sampling:
            
            mu  = self.h5_handler.get(self.col_name)[index]
            std = self.h5_handler.get(self.colstd_name)[index]

            img = self.sample(mu, std)
        else:
            
            img = self.h5_handler.get(self.col_name)[index]
            
        img = (img - self.global_min) / (self.global_max - self.global_min)
        img = np.clip(img, 0, 1)

        
        item = torch.from_numpy(img).type(torch.FloatTensor)
        # item = item.permute(3,0,1,2)
        
        return item