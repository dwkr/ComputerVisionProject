from torch.utils import data
import numpy as np

class GTADataset(data.Dataset):
    """GTA dataset"""
    
    def __init__(self, set, offset):
        self.set = set
        self.offset = offset
        
    def __len__(self):
        return len(self.set) - self.offset
        
    def __getitem__(self, idx):
        X1 = np.array(self.set[self.offset + idx][0], dtype=np.float32).reshape(3,299,299)
        X2 = np.array(self.set[self.offset + idx][2], dtype=np.float32).reshape(1,64,64)
        X3 = np.array(self.set[self.offset + idx][3], dtype=np.float32).reshape(1,64,64)
        X4 = np.array(self.set[self.offset + idx][4], dtype=np.float32).reshape(100,5)
        Y = np.array(self.set[self.offset + idx][1], dtype=np.float32).reshape(5)
    
        #X1 = np.array([i[0] for i in data], dtype=np.float32).reshape(n_batches,batch_size,3,299,299)
        #X2 = np.array([i[2] for i in data], dtype=np.float32).reshape(n_batches,batch_size,1,64,64)
        #X3 = np.array([i[3] for i in data], dtype=np.float32).reshape(n_batches,batch_size,1,64,64)
        #X4 = np.array([i[4] for i in data], dtype=np.float32).reshape(n_batches,batch_size,100,5)
        #Y = np.array([i[1] for i in data]).reshape(n_batches, batch_size, 5)
        return [X1, X2, X3, X4], Y