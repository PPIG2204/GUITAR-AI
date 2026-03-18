# src/model.py
import torch
import torch.nn as nn
import config

class GuitarTranscriberCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((4, 1)), 
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((4, 1)),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )
        
        # ---------------------------------------------------------
        # THE DYNAMIC MATH ENGINE (Auto-calculating input size)
        # ---------------------------------------------------------
        if config.FEATURE_TYPE == 'CQT':
            input_height = config.CQT_BINS
        elif config.FEATURE_TYPE == 'MEL':
            input_height = config.N_MELS
        elif config.FEATURE_TYPE == 'STFT':
            input_height = (config.N_FFT // 2) + 1

        # Use a dummy tensor to get the exact output shape of conv_block
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_height, config.CONTEXT_LENGTH)
            dummy_output = self.conv_block(dummy_input)
            # Flatten everything except batch and time (the time dimension is config.CONTEXT_LENGTH)
            linear_input_size = dummy_output.shape[1] * dummy_output.shape[2]

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(linear_input_size, 6 * 21)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)
        return self.classifier(x).view(x.size(0), x.size(1), 6, 21)
