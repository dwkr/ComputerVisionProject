import torch
import torch.nn as nn
from R2plus1D import R2Plus1DNet
from torchvision import models


class VideoR2Plus1D(nn.Module):
    """docstring for R2+1 D Net"""

    def __init__(self, num_inputs, num_outputs):
        super(VideoR2Plus1D, self).__init__()
        self.model = R2Plus1DNet(layer_sizes=[2, 2, 2, 2])
        self.fc = nn.Linear(in_features=512, out_features=num_outputs, bias=True)

    def forward(self, x):
        batch_size, clip_len, img_channels, height, width = x.shape
        x = x.view(batch_size, img_channels, clip_len, height, width)
        x = self.model(x)
        return self.fc(x)





class VideoResNet50_LSTM(nn.Module):
    """docstring for ImageResNet50"""

    def __init__(self, num_inputs, num_outputs):
        super(VideoResNet50_LSTM, self).__init__()

        self.model = models.resnet50()
        self.model.conv1 = nn.Conv2d(num_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(in_features=2048, out_features=num_outputs, bias=True)

        self.rnn_cell_size = num_outputs // 2

        self.rnn = nn.LSTM(input_size=num_outputs, hidden_size=512, num_layers=1, bias=True, batch_first=True,
                           dropout=0.5, bidirectional=False)

        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.rnn_cell_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 5))

    def forward(self, x):
        cnn_output = None

        batch_size, clip_len, img_channels, height, width = x.shape
        x = x.view(clip_len, batch_size, img_channels, height, width)

        for xi in x:
            if cnn_output is None:
                cnn_output = self.model(xi).unsqueeze_(1)
            else:
                cnn_output = torch.cat((cnn_output, self.model(xi).unsqueeze_(1)), 1)

        batch_size = cnn_output.shape[0]
        cell_size = cnn_output.shape[-1] // 2
        output, (h, c) = self.rnn(cnn_output, (torch.zeros(1, batch_size, cell_size),
                                               torch.zeros(1, batch_size, cell_size)))
        h = h.squeeze_(0)
        c = c.squeeze_(0)
        lstm_states = torch.cat((h, c), -1)

        prev_clips = output[:, 0:-1, :]
        output = torch.cat((output[:, -1, :], lstm_states), -1)

        batch_size, prev_clip_len, num_features = prev_clips.shape
        prev_clips = prev_clips.view(prev_clip_len, batch_size, num_features)

        prev_outputs = None
        for prev_clip in prev_clips:
            out = self.fc(prev_clip).unsqueeze_(1)
            if prev_outputs is None:
                prev_outputs = out
            else:
                prev_outputs = torch.cat((prev_outputs, out), 1)

        return prev_outputs, output

    def name(self):
        return "VideoResNet50_LSTM"


class VideoSqueeze_LSTM(nn.Module):
    """docstring for ImageResNet50"""

    def __init__(self, num_inputs, num_outputs):
        super(VideoSqueeze_LSTM, self).__init__()

        self.model = models.squeezenet1_1(num_classes=num_outputs)

        self.rnn_cell_size = 512

        self.rnn = nn.LSTM(input_size=num_outputs, hidden_size=self.rnn_cell_size, num_layers=1, bias=True,
                           batch_first=True, dropout=0.5, bidirectional=False)

        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.rnn_cell_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 5))

    def forward(self, x):
        cnn_output = None
        batch_size, clip_len, img_channels, height, width = x.shape
        x = x.view(clip_len, batch_size, img_channels, height, width)

        for xi in x:
            if cnn_output is None:
                cnn_output = self.model(xi).unsqueeze_(1)
            else:
                cnn_output = torch.cat((cnn_output, self.model(xi).unsqueeze_(1)), 1)

        batch_size = cnn_output.shape[0]
        cell_size = cnn_output.shape[-1] // 2
        output, (h, c) = self.rnn(cnn_output, (torch.zeros(1, batch_size, cell_size),
                                               torch.zeros(1, batch_size, cell_size)))
        h = h.squeeze_(0)
        c = c.squeeze_(0)
        lstm_states = torch.cat((h, c), -1)

        prev_clips = output[:, 0:-1, :]
        output = torch.cat((output[:, -1, :], lstm_states), -1)

        batch_size, prev_clip_len, num_features = prev_clips.shape
        prev_clips = prev_clips.contiguous().view(prev_clip_len, batch_size, num_features)

        prev_outputs = None
        for prev_clip in prev_clips:
            out = self.fc(prev_clip).unsqueeze_(1)
            if prev_outputs is None:
                prev_outputs = out
            else:
                prev_outputs = torch.cat((prev_outputs, out), 1)

        return prev_outputs, output

    def name(self):
        return "VideoSqueeze_LSTM"
