import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

def compare_metrics(results, metric_name, title, filename):
    """Сравнение метрик производительности моделей"""
    plt.figure(figsize=(10, 6))
    for model_name, result in results.items():
        plt.plot(result[metric_name], label=model_name)
    plt.title(title)
    plt.xlabel('Эпоха')
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_confusion_matrices(results, classes, filename_prefix):
    """Визуализация матриц ошибок для всех моделей"""
    for model_name, result in results.items():
        cm = confusion_matrix(result['true_labels'], result['predictions'])
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(f'Матрица ошибок: {model_name}')
        plt.xlabel('Предсказанный класс')
        plt.ylabel('Истинный класс')
        plt.savefig(f'{filename_prefix}_{model_name.lower().replace(" ", "_")}.png')
        plt.close()

def save_comparison_table(results, filename):
    """Сохранение таблицы сравнения метрик"""
    data = {
        'Модель': [],
        'Точность на тесте (%)': [],
        'Время обучения (с)': [],
        'Количество параметров': [],
        'Средняя норма градиента': []
    }
    for model_name, result in results.items():
        data['Модель'].append(model_name)
        data['Точность на тесте (%)'].append(result['test_accs'][-1] if 'test_accs' in result else 0)
        data['Время обучения (с)'].append(result['training_time'] if 'training_time' in result else 0)
        data['Количество параметров'].append(result['num_params'] if 'num_params' in result else 0)
        data['Средняя норма градиента'].append(np.mean(result['grad_norms']) if 'grad_norms' in result else 0)
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    return df

def analyze_gradients(results, filename):
    """Анализ распределения норм градиентов"""
    plt.figure(figsize=(10, 6))
    for model_name, result in results.items():
        if 'grad_norms' in result:
            sns.histplot(result['grad_norms'], label=model_name, kde=True)
    plt.title('Распределение норм градиентов')
    plt.xlabel('Норма градиента')
    plt.ylabel('Частота')
    plt.legend()
    plt.savefig(filename)
    plt.close()