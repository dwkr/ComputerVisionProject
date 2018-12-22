from torch.utils import data
import numpy as np
import glob
from extract_features import getFeatures
from find_stats import findStats
import torch


class GTADataSetSequential(data.Dataset):
    """GTA Clip Data Set Genrator"""

    def __init__(self, path, stats, logging, norm = True):
        self.path = path
        self.files = glob.glob(path)
        self.files.sort()
        self.file_number = -1
        self.file_offset = 0
        self.norm_stats = stats
        self.logging = logging
        self.norm = norm

        self.logging.info("Number of files in path {} = {}".format(self.path, len(self.files)))

    def find_distribution(self):
        feature_count = np.zeros(5)
        for feature in self.features:
            feature_count = feature_count + feature[1]
        return feature_count


    def load_file(self):
        file_data = np.load(self.files[self.file_number])
        self.logging.info("Reading from file :  {}".format(self.files[self.file_number]))
        self.features = getFeatures(file_data)
        self.logging.info("Feature Distribution : {}".format(self.find_distribution()))


    def normalize(self, X, s):
        X = X / 255.0

        # for channel in range(X.shape[0]):
        #     X[channel, :, :] = X[channel, :, :] - np.array(self.norm_stats[s]['mean'][channel], dtype=np.float32)
        #     X[channel, :, :] /= np.array(self.norm_stats[s]['std'][channel], dtype=np.float32)

        return X

    def __getitem__(self, idx):
        
        file_number = idx//500
        if self.file_number != file_number:
            self.file_number = file_number
            self.load_file()



        X1 = np.array(self.features[idx%500][0], dtype=np.float32).reshape(3,112,112)
        X2 = np.array(self.features[idx%500][2], dtype=np.float32).reshape(1, 64, 64)
        X3 = np.array(self.features[idx%500][3], dtype=np.float32).reshape(1, 64, 64)
        X4 = np.array(self.features[idx%500][4], dtype=np.float32).reshape(100, 5)
        Y = np.array(self.features[idx%500][1], dtype=np.float32).reshape(5)

        if self.norm:
            return [self.normalize(X1,'X1'), self.normalize(X2,'X2'), self.normalize(X3,'X3'), X4], Y
        else:
            return  [X1, X2, X3, X4], Y

    def __len__(self):
        num_files = len(self.files)
        return len(np.load(self.files[0])) * (num_files-1) + len(np.load(self.files[num_files-1]))


