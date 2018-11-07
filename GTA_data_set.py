from torch.utils import data
import numpy as np

class GTADataset(data.Dataset):
    """GTA dataset"""
    
    def __init__(self, set, offset, ratio, stats, normalize = False):
        self.set = set
        self.offset = offset
        self.length = int(ratio * len(self.set))
        self.norm_stats = stats
        self.normalize = normalize
        
    def __len__(self):
        return self.length
        
    def normalize(X, s):
        X = X/255
        
        for channel in range(X1.shape[0]):
            X = X[channel,:,:] - stats[s]['mean'][channel]
            X /= stats[s]['std'][channel]

        return X
    
    def __getitem__(self, idx):
        X1 = np.array(self.set[self.offset + idx][0], dtype=np.float32).reshape(3,299,299)
        X2 = np.array(self.set[self.offset + idx][2], dtype=np.float32).reshape(1,64,64)
        X3 = np.array(self.set[self.offset + idx][3], dtype=np.float32).reshape(1,64,64)
        X4 = np.array(self.set[self.offset + idx][4], dtype=np.float32).reshape(100,5)
        Y = np.array(self.set[self.offset + idx][1], dtype=np.float32).reshape(5)

        return [normalize(X1,'X1'), normalize(X2,'X2'), normalize(X3,'X3'), X4], Y if self.normalize
                else [X1, X2, X3, X4], Y