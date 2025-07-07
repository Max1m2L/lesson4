import torch
import torch.nn as nn

class CNNKernelSize(nn.Module):
    """CNN с разными размерами ядер свертки"""
    def __init__(self, input_channels, num_classes, kernel_sizes, out_channels):
        super(CNNKernelSize, self).__init__()
        layers = []
        in_channels = input_channels
        for k, out_c in zip(kernel_sizes, out_channels):
            layers.extend([
                nn.Conv2d(in_channels, out_c, kernel_size=k, padding=k//2),
                nn.ReLU(),
                nn.BatchNorm2d(out_c)
            ])
            in_channels = out_c
        layers.append(nn.MaxPool2d(2, 2))
        self.features = nn.Sequential(*layers)
        # Для CIFAR-10: после MaxPool2d(2, 2) размер 32x32 -> 16x16
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_channels[-1] * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ResNetBlock(nn.Module):
    """Residual блок"""
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class CNNDepth(nn.Module):
    """CNN с разной глубиной"""
    def __init__(self, input_channels, num_classes, num_conv_layers, use_residual=False):
        super(CNNDepth, self).__init__()
        layers = []
        in_channels = input_channels
        out_channels = 32
        pool_count = 0
        for i in range(num_conv_layers):
            if use_residual and i % 2 == 0 and i > 0:  # Используем ResNetBlock после первого слоя
                layers.append(ResNetBlock(in_channels, out_channels))
            else:
                layers.extend([
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(out_channels)
                ])
            in_channels = out_channels
            if (i + 1) % 2 == 0:  # Добавляем MaxPool2d каждые два слоя
                layers.append(nn.MaxPool2d(2, 2))
                pool_count += 1
                out_channels *= 2
        self.features = nn.Sequential(*layers)
        # Для CIFAR-10: начальный размер 32x32
        # После pool_count MaxPool2d(2, 2) размер уменьшается в 2^pool_count
        spatial_size = 32 // (2 ** pool_count)  # 32 -> 16 -> 8 (если pool_count=2)
        final_channels = out_channels if pool_count == 0 else out_channels // 2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_channels * spatial_size * spatial_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x