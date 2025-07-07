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
from models.cnn_models import CNNKernelSize, CNNDepth
from utils.training_utils import train_model, evaluate_model
from utils.visualization_utils import plot_training_curves, visualize_feature_maps

# Создание необходимых директорий
os.makedirs('results/architecture_analysis', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Настройка логирования
logging.basicConfig(filename='results/architecture_analysis/experiment.log', level=logging.INFO,
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

def analyze_kernel_size():
    """Анализ влияния размера ядра свертки"""
    trainloader, testloader = setup_data()
    configs = [
        {'kernel_sizes': [3, 3], 'out_channels': [32, 64], 'name': 'Ядра 3x3'},
        {'kernel_sizes': [5, 5], 'out_channels': [32, 64], 'name': 'Ядра 5x5'},
        {'kernel_sizes': [7, 7], 'out_channels': [32, 64], 'name': 'Ядра 7x7'}
    ]
    
    results = {}
    for config in configs:
        logging.info(f'Обучение модели с {config["name"]}')
        model = CNNKernelSize(input_channels=3, num_classes=10, 
                            kernel_sizes=config['kernel_sizes'], 
                            out_channels=config['out_channels']).to(
                            torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        start_time = time.time()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        train_losses, test_losses, train_accs, test_accs = train_model(
            model, trainloader, testloader, criterion, optimizer, num_epochs=10
        )
        end_time = time.time()
        
        # Подсчет параметров
        num_params = sum(p.numel() for p in model.parameters())
        
        results[config['name']] = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accs': train_accs,
            'test_accs': test_accs,
            'training_time': end_time - start_time,
            'num_params': num_params
        }
        logging.info(f'{config["name"]}: Точность = {test_accs[-1]:.4f}, Время = {end_time - start_time:.2f}с, Параметры = {num_params}')
        
        # Визуализация кривых обучения
        plot_training_curves(train_losses, test_losses, train_accs, test_accs, 
                           f'plots/kernel_size_{config["name"].lower().replace(" ", "_")}_curves.png')
    
    # Сохранение результатов
    with open('results/architecture_analysis/kernel_size_results.txt', 'w') as f:
        for name, result in results.items():
            f.write(f'{name}:\n')
            f.write(f'Точность на тесте: {result["test_accs"][-1]:.4f}\n')
            f.write(f'Время обучения: {result["training_time"]:.2f}с\n')
            f.write(f'Количество параметров: {result["num_params"]}\n\n')

def analyze_depth():
    """Анализ влияния глубины сети"""
    trainloader, testloader = setup_data()
    configs = [
        {'num_conv_layers': 2, 'use_residual': False, 'name': '2 слоя без Residual'},
        {'num_conv_layers': 4, 'use_residual': False, 'name': '4 слоя без Residual'},
        {'num_conv_layers': 2, 'use_residual': True, 'name': '2 слоя с Residual'},
        {'num_conv_layers': 4, 'use_residual': True, 'name': '4 слоя с Residual'}
    ]
    
    results = {}
    for config in configs:
        logging.info(f'Обучение модели с {config["name"]}')
        model = CNNDepth(input_channels=3, num_classes=10, 
                        num_conv_layers=config['num_conv_layers'], 
                        use_residual=config['use_residual']).to(
                        torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        start_time = time.time()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        train_losses, test_losses, train_accs, test_accs = train_model(
            model, trainloader, testloader, criterion, optimizer, num_epochs=10
        )
        end_time = time.time()
        
        # Подсчет параметров
        num_params = sum(p.numel() for p in model.parameters())
        
        # Визуализация feature maps
        sample_image, _ = next(iter(testloader))
        sample_image = sample_image[0:1].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        visualize_feature_maps(model, sample_image, 
                             f'plots/depth_{config["name"].lower().replace(" ", "_")}_feature_maps.png')
        
        results[config['name']] = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accs': train_accs,
            'test_accs': test_accs,
            'training_time': end_time - start_time,
            'num_params': num_params
        }
        logging.info(f'{config["name"]}: Точность = {test_accs[-1]:.4f}, Время = {end_time - start_time:.2f}с, Параметры = {num_params}')
    
    # Сохранение результатов
    with open('results/architecture_analysis/depth_results.txt', 'w') as f:
        for name, result in results.items():
            f.write(f'{name}:\n')
            f.write(f'Точность на тесте: {result["test_accs"][-1]:.4f}\n')
            f.write(f'Время обучения: {result["training_time"]:.2f}с\n')
            f.write(f'Количество параметров: {result["num_params"]}\n\n')

if __name__ == '__main__':
    analyze_kernel_size()
    analyze_depth()