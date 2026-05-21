import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt
    import pandas as pd

    return StandardScaler, nn, np, optim, plt, torch, train_test_split


@app.cell
def _(StandardScaler, np, torch, train_test_split):
    X = np.loadtxt('Lab5/dataIn.txt')
    y = np.loadtxt('Lab5/dataOut.txt')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_classic = X_train
    y_train_classic = np.argmax(y_train, axis=1)

    X_test_classic = X_test
    y_test_classic = np.argmax(y_test, axis=1)

    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    return X_test, X_train, y_test, y_train


@app.cell
def _(nn, optim):
    class ElectNet(nn.Module):
        def __init__(self):
            super(ElectNet, self).__init__()
            self.hidden = nn.Linear(12, 10)
            self.activation = nn.Sigmoid()
            self.output = nn.Linear(10, 2)
            self.final_act = nn.Sigmoid()

        def forward(self, x):
            x = self.hidden(x)
            x = self.activation(x)
            x = self.output(x)
            x = self.final_act(x)
            return x

    model = ElectNet()
    crit = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    return crit, model, optimizer


@app.cell
def _(X_train, crit, model, optimizer, y_train):
    epochs = 200
    history_loss = []
    for epoch in range(epochs):
        outputs = model(X_train)
        loss = crit(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history_loss.append(loss.item())
    return (history_loss,)


@app.cell
def _(X_test, model, torch, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        correct = (predictions.argmax(1) == y_test.argmax(1)).sum().item()
        accuracy = correct / y_test.size(0)
        print(f'Multi-Layer Perceptron accuracy: {accuracy * 100:.2f}%')
    return


@app.cell
def _(history_loss, plt):
    plt.figure(figsize=(10, 5))
    plt.plot(history_loss)
    plt.title('График обучения модели (Loss)')
    plt.xlabel('Эпоха')
    plt.ylabel('Ошибка')
    plt.grid(True)
    plt.show()
    return


@app.cell
def _(X_test, model, plt, torch, y_test):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    model.eval()
    with torch.no_grad():
        raw_preds = model(X_test)
        y_pred = raw_preds.argmax(1).numpy()
        y_true = y_test.argmax(1).numpy()

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Правящая', 'Оппозиция'], 
                yticklabels=['Правящая', 'Оппозиция'])
    plt.title('Матрица ошибок (Результаты предсказаний)')
    plt.ylabel('Реальный результат')
    plt.xlabel('Предсказание нейросети')
    plt.show()
    return


if __name__ == "__main__":
    app.run()
