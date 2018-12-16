from torch.utils import data
import numpy as np

class GTADataset(data.Dataset):
    """GTA dataset"""
    
    def __init__(self, set, offset, ratio, stats, norm = False):
        self.set = set
        self.offset = offset
        self.length = int(ratio * len(self.set))
        self.norm_stats = stats
        self.norm = norm
        
    def __len__(self):
        return self.length
        
    def normalize(self, X, s):
        X = X/255
        
        for channel in range(X.shape[0]):
            X[channel,:,:] = X[channel,:,:] - np.array(self.norm_stats[s]['mean'][channel], dtype=np.float32)
            X[channel,:,:] /= np.array(self.norm_stats[s]['std'][channel], dtype=np.float32)

        return X
    
    def __getitem__(self, idx):
        X1 = np.array(self.set[self.offset + idx][0], dtype=np.float32).reshape(3,224,224)
        X2 = np.array(self.set[self.offset + idx][2], dtype=np.float32).reshape(1,64,64)
        X3 = np.array(self.set[self.offset + idx][3], dtype=np.float32).reshape(1,64,64)
        X4 = np.array(self.set[self.offset + idx][4], dtype=np.float32).reshape(100,5)
        Y = np.array(self.set[self.offset + idx][1], dtype=np.float32).reshape(5)

        if self.norm:
            return [self.normalize(X1,'X1'), self.normalize(X2,'X2'), self.normalize(X3,'X3'), X4], Y
        else:
            return  [X1, X2, X3, X4], Y