import torch
import time

def train_model(model, trainloader, testloader, criterion, optimizer, num_epochs):
    """Обучение модели с логированием потерь и точности"""
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(trainloader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Исправлено: распаковка всех возвращаемых значений
        test_loss, test_acc, _, _, _ = evaluate_model(model, testloader, criterion)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
    return train_losses, test_losses, train_accs, test_accs

def evaluate_model(model, testloader, criterion):
    """Оценка модели на тестовом наборе"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    start_time = time.time()
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    inference_time = time.time() - start_time
    test_loss = test_loss / len(testloader)
    test_acc = 100 * correct / total
    return test_loss, test_acc, inference_time, predictions, true_labels