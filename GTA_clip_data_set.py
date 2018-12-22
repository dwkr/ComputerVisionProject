from torch.utils import data
import numpy as np
import glob
from extract_features import getFeatures
from find_stats import findStats
import torch


class GTAClipDataSet(data.Dataset):
    """GTA Clip Data Set Genrator"""

    def __init__(self, path, clip_len, stats, logging, files):
        self.train_path = path
        self.files = files
        self.files.sort()
        self.file_number = -1
        self.file_offset = clip_len
        self.clip_len = clip_len
        self.norm_stats = stats
        self.logging = logging

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
        X = X / 255

        for channel in range(X.shape[0]):
            X[channel, :, :] = X[channel, :, :] - np.array(self.norm_stats[s]['mean'][channel], dtype=np.float32)
            X[channel, :, :] /= np.array(self.norm_stats[s]['std'][channel], dtype=np.float32)

        return X

    def __getitem__(self, idx):

        file_number = idx//(500 - self.clip_len + 1)
        if self.file_number != file_number:
            self.file_number = file_number
            self.load_file()

        self.file_offset = idx % (500 - self.clip_len + 1)

        clip = []
        Y = []
        for i in range(self.file_offset, self.file_offset+self.clip_len):
            clip.append(self.normalize(self.features[i][0].reshape(3, 224, 224), 'X1'))
            Y.append(self.features[i][1])

        clip = np.array(clip, dtype=np.float32).reshape(self.clip_len, 3, 224, 224)
        Y = np.array(Y).reshape(self.clip_len, 5)

        X2 = np.array(self.features[self.file_offset + self.clip_len - 1][2], dtype=np.float32).reshape(1, 64, 64)
        X3 = np.array(self.features[self.file_offset + self.clip_len - 1][3], dtype=np.float32).reshape(1, 64, 64)
        X4 = np.array(self.features[self.file_offset + self.clip_len - 1][4], dtype=np.float32).reshape(100, 5)

        return [clip, X2, X3, X4], Y

    def __len__(self):
        return len(self.files) * (500 - self.clip_len + 1)


