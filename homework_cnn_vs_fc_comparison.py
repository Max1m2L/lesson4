import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import logging
import time
import os
from models.fc_models import FullyConnectedMNIST, FullyConnectedCIFAR
from models.cnn_models import SimpleCNN, ResNetBlockCNN
from utils.training_utils import train_model, evaluate_model
from utils.visualization_utils import plot_training_curves, plot_confusion_matrix

# Создание необходимых директорий
os.makedirs('results/mnist_comparison', exist_ok=True)
os.makedirs('results/cifar_comparison', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Настройка логирования
logging.basicConfig(filename='results/mnist_comparison/experiment.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def setup_data(dataset_name='MNIST'):
    """Настройка загрузчиков данных для MNIST или CIFAR-10"""
    os.makedirs('data', exist_ok=True)  # Создание папки для датасетов
    if dataset_name == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    else:  # CIFAR-10
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
    return trainloader, testloader

def compare_models_mnist():
    """Сравнение моделей на MNIST"""
    trainloader, testloader = setup_data('MNIST')
    models = {
        'Полносвязная сеть': FullyConnectedMNIST(),
        'Простая CNN': SimpleCNN(input_channels=1, num_classes=10),
        'CNN с Residual': ResNetBlockCNN(input_channels=1, num_classes=10)
    }
    
    results = {}
    for name, model in models.items():
        logging.info(f'Обучение модели: {name}')
        start_time = time.time()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        train_losses, test_losses, train_accs, test_accs = train_model(
            model, trainloader, testloader, criterion, optimizer, num_epochs=10
        )
        end_time = time.time()
        
        # Оценка времени инференса
        test_loss, test_acc, inference_time, predictions, true_labels = evaluate_model(model, testloader, criterion)
        
        # Подсчет параметров
        num_params = sum(p.numel() for p in model.parameters())
        
        results[name] = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accs': train_accs,
            'test_accs': test_accs,
            'training_time': end_time - start_time,
            'inference_time': inference_time,
            'num_params': num_params,
            'predictions': predictions,
            'true_labels': true_labels
        }
        logging.info(f'{name}: Точность на тесте = {test_accs[-1]:.4f}, Время обучения = {end_time - start_time:.2f}с, Параметры = {num_params}')
        
        # Визуализация кривых обучения
        plot_training_curves(train_losses, test_losses, train_accs, test_accs, f'plots/mnist_{name.lower().replace(" ", "_")}_curves.png')
    
    # Сохранение результатов
    with open('results/mnist_comparison/results.txt', 'w') as f:
        for name, result in results.items():
            f.write(f'{name}:\n')
            f.write(f'Точность на тесте: {result["test_accs"][-1]:.4f}\n')
            f.write(f'Время обучения: {result["training_time"]:.2f}с\n')
            f.write(f'Время инференса: {result["inference_time"]:.4f}с\n')
            f.write(f'Количество параметров: {result["num_params"]}\n\n')

def compare_models_cifar():
    """Сравнение моделей на CIFAR-10"""
    trainloader, testloader = setup_data('CIFAR-10')
    models = {
        'Полносвязная сеть': FullyConnectedCIFAR(),
        'CNN с Residual': ResNetBlockCNN(input_channels=3, num_classes=10),
        'CNN с регуляризацией': ResNetBlockCNN(input_channels=3, num_classes=10, dropout=0.5)
    }
    
    results = {}
    for name, model in models.items():
        logging.info(f'Обучение модели: {name}')
        start_time = time.time()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        train_losses, test_losses, train_accs, test_accs = train_model(
            model, trainloader, testloader, criterion, optimizer, num_epochs=20
        )
        end_time = time.time()
        
        # Оценка и визуализация confusion matrix
        test_loss, test_acc, inference_time, predictions, true_labels = evaluate_model(model, testloader, criterion)
        cm = confusion_matrix(true_labels, predictions)
        plot_confusion_matrix(cm, classes=[str(i) for i in range(10)], 
                            filename=f'plots/cifar_{name.lower().replace(" ", "_")}_cm.png')
        
        # Анализ градиентов
        grad_norms = []
        for param in model.parameters():
            if param.grad is not None:
                grad_norms.append(torch.norm(param.grad).item())
        
        results[name] = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accs': train_accs,
            'test_accs': test_accs,
            'training_time': end_time - start_time,
            'grad_norms': grad_norms,
            'predictions': predictions,
            'true_labels': true_labels
        }
        logging.info(f'{name}: Точность на тесте = {test_accs[-1]:.4f}, Время обучения = {end_time - start_time:.2f}с')
    
    # Визуализация кривых обучения
    for name, result in results.items():
        plot_training_curves(result['train_losses'], result['test_losses'], 
                           result['train_accs'], result['test_accs'], 
                           f'plots/cifar_{name.lower().replace(" ", "_")}_curves.png')

if __name__ == '__main__':
    compare_models_mnist()
    compare_models_cifar()
    