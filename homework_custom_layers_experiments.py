import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import logging
import time
import os
from models.custom_layers import CustomConv2d, CustomAttention, CustomActivation, CustomPooling
from utils.training_utils import train_model, evaluate_model
from utils.visualization_utils import plot_training_curves

# Создание необходимых директорий
os.makedirs('results/custom_layers_experiments', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Настройка логирования
logging.basicConfig(filename='results/custom_layers_experiments/experiment.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def setup_data():
    """Настройка загрузчиков данных для CIFAR-10"""
    os.makedirs('data', exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
    return trainloader, testloader

class CustomCNN(nn.Module):
    """CNN с кастомными слоями"""
    def __init__(self, use_custom_conv=False, use_attention=False, use_custom_act=False, use_custom_pool=False):
        super(CustomCNN, self).__init__()
        conv_layer = CustomConv2d(3, 64, kernel_size=3, padding=1) if use_custom_conv else nn.Conv2d(3, 64, kernel_size=3, padding=1)
        act_layer = CustomActivation(0.1) if use_custom_act else nn.ReLU()
        pool_layer = CustomPooling(output_size=16) if use_custom_pool else nn.MaxPool2d(2, 2)  # Изменено на output_size=16
        layers = [
            conv_layer,
            act_layer,
            nn.BatchNorm2d(64),
            pool_layer
        ]
        if use_attention:
            layers.append(CustomAttention(64))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),  # Исправлено: 16x16 вместо 8x8
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class BasicResidualBlock(nn.Module):
    """Базовый Residual блок"""
    def __init__(self, in_channels, out_channels):
        super(BasicResidualBlock, self).__init__()
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

class BottleneckResidualBlock(nn.Module):
    """Bottleneck Residual блок"""
    def __init__(self, in_channels, out_channels, reduction=4):
        super(BottleneckResidualBlock, self).__init__()
        bottleneck_channels = out_channels // reduction
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class WideResidualBlock(nn.Module):
    """Wide Residual блок"""
    def __init__(self, in_channels, out_channels, widen_factor=2):
        super(WideResidualBlock, self).__init__()
        wide_channels = out_channels * widen_factor
        self.conv1 = nn.Conv2d(in_channels, wide_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(wide_channels)
        self.conv2 = nn.Conv2d(wide_channels, out_channels, kernel_size=3, padding=1)
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

class ResidualCNN(nn.Module):
    """CNN с Residual блоками"""
    def __init__(self, block_type, num_blocks=2):
        super(ResidualCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        in_channels = 64
        for _ in range(num_blocks):
            self.features.add_module(f"block_{_}", block_type(in_channels, 64))
        self.features.add_module("pool", nn.MaxPool2d(2, 2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),  # 32x32 -> 16x16 после MaxPool2d(2, 2)
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def experiment_custom_layers():
    """Эксперименты с кастомными слоями"""
    trainloader, testloader = setup_data()
    configs = [
        {'name': 'Базовая CNN', 'use_custom_conv': False, 'use_attention': False, 'use_custom_act': False, 'use_custom_pool': False},
        {'name': 'Кастомная свертка', 'use_custom_conv': True, 'use_attention': False, 'use_custom_act': False, 'use_custom_pool': False},
        {'name': 'Attention', 'use_custom_conv': False, 'use_attention': True, 'use_custom_act': False, 'use_custom_pool': False},
        {'name': 'Кастомная активация', 'use_custom_conv': False, 'use_attention': False, 'use_custom_act': True, 'use_custom_pool': False},
        {'name': 'Кастомный пуллинг', 'use_custom_conv': False, 'use_attention': False, 'use_custom_act': False, 'use_custom_pool': True}
    ]
    
    results = {}
    for config in configs:
        name = config.pop('name')  # Удаляем 'name' из config
        logging.info(f'Обучение модели: {name}')
        model = CustomCNN(**config).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        start_time = time.time()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        train_losses, test_losses, train_accs, test_accs = train_model(
            model, trainloader, testloader, criterion, optimizer, num_epochs=10
        )
        end_time = time.time()
        
        # Подсчет параметров
        num_params = sum(p.numel() for p in model.parameters())
        
        # Оценка стабильности (дисперсия потерь)
        stability = np.var(test_losses)
        
        results[name] = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accs': train_accs,
            'test_accs': test_accs,
            'training_time': end_time - start_time,
            'num_params': num_params,
            'stability': stability
        }
        logging.info(f'{name}: Точность = {test_accs[-1]:.4f}, Время = {end_time - start_time:.2f}с, Параметры = {num_params}, Стабильность = {stability:.4f}')
        
        # Визуализация кривых обучения
        plot_training_curves(train_losses, test_losses, train_accs, test_accs, 
                           f'plots/custom_layers_{name.lower().replace(" ", "_")}_curves.png')
    
    # Сохранение результатов
    with open('results/custom_layers_experiments/custom_layers_results.txt', 'w') as f:
        for name, result in results.items():
            f.write(f'{name}:\n')
            f.write(f'Точность на тесте: {result["test_accs"][-1]:.4f}\n')
            f.write(f'Время обучения: {result["training_time"]:.2f}с\n')
            f.write(f'Количество параметров: {result["num_params"]}\n')
            f.write(f'Стабильность (дисперсия потерь): {result["stability"]:.4f}\n\n')

def experiment_residual_blocks():
    """Эксперименты с Residual блоками"""
    trainloader, testloader = setup_data()
    configs = [
        {'name': 'Базовый Residual', 'block_type': BasicResidualBlock},
        {'name': 'Bottleneck Residual', 'block_type': BottleneckResidualBlock},
        {'name': 'Wide Residual', 'block_type': WideResidualBlock}
    ]
    
    results = {}
    for config in configs:
        name = config['name']
        logging.info(f'Обучение модели с {name}')
        model = ResidualCNN(config['block_type']).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        start_time = time.time()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        train_losses, test_losses, train_accs, test_accs = train_model(
            model, trainloader, testloader, criterion, optimizer, num_epochs=10
        )
        end_time = time.time()
        
        # Подсчет параметров
        num_params = sum(p.numel() for p in model.parameters())
        
        # Оценка стабильности (дисперсия потерь)
        stability = np.var(test_losses)
        
        results[name] = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accs': train_accs,
            'test_accs': test_accs,
            'training_time': end_time - start_time,
            'num_params': num_params,
            'stability': stability
        }
        logging.info(f'{name}: Точность = {test_accs[-1]:.4f}, Время = {end_time - start_time:.2f}с, Параметры = {num_params}, Стабильность = {stability:.4f}')
        
        # Визуализация кривых обучения
        plot_training_curves(train_losses, test_losses, train_accs, test_accs, 
                           f'plots/residual_{name.lower().replace(" ", "_")}_curves.png')
    
    # Сохранение результатов
    with open('results/custom_layers_experiments/residual_results.txt', 'w') as f:
        for name, result in results.items():
            f.write(f'{name}:\n')
            f.write(f'Точность на тесте: {result["test_accs"][-1]:.4f}\n')
            f.write(f'Время обучения: {result["training_time"]:.2f}с\n')
            f.write(f'Количество параметров: {result["num_params"]}\n')
            f.write(f'Стабильность (дисперсия потерь): {result["stability"]:.4f}\n\n')

if __name__ == '__main__':
    experiment_custom_layers()
    experiment_residual_blocks()