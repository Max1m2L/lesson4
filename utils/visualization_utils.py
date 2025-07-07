import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn

def plot_training_curves(train_losses, test_losses, train_accs, test_accs, filename):
    """Визуализация кривых обучения"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Обучающая потеря')
    plt.plot(test_losses, label='Тестовая потеря')
    plt.title('Кривые потерь')
    plt.xlabel('Эпоха')
    plt.ylabel('Потеря')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Обучающая точность')
    plt.plot(test_accs, label='Тестовая точность')
    plt.title('Кривые точности')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_confusion_matrix(cm, classes, filename):
    """Визуализация матрицы ошибок"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)  # Исправлено: sns.heatmap
    plt.title('Матрица ошибок')
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.savefig(filename)
    plt.close()

def visualize_activations(model, input_image, filename):
    """Визуализация активаций первого слоя"""
    model.eval()
    with torch.no_grad():
        activations = model.features[0](input_image)
        activations = activations.squeeze().cpu().numpy()
    
    plt.figure(figsize=(15, 5))
    for i in range(min(8, activations.shape[0])):
        plt.subplot(2, 4, i+1)
        plt.imshow(activations[i], cmap='viridis')
        plt.axis('off')
        plt.title(f'Фильтр {i+1}')
    plt.savefig(filename)
    plt.close()

def visualize_feature_maps(model, input_image, filename):
    """Визуализация feature maps всех слоев"""
    model.eval()
    feature_maps = []
    x = input_image
    with torch.no_grad():
        for layer in model.features:
            x = layer(x)
            if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU)):
                feature_maps.append(x.squeeze().cpu().numpy())
    
    plt.figure(figsize=(15, 10))
    for i, fmap in enumerate(feature_maps[:4]):  # Ограничимся первыми 4 слоями
        plt.subplot(2, 2, i+1)
        plt.imshow(fmap[0], cmap='viridis')  # Первый канал
        plt.axis('off')
        plt.title(f'Слой {i+1}')
    plt.savefig(filename)
    plt.close()