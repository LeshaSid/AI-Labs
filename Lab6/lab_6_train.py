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
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    print("Версия PyTorch:", torch.__version__)
    print("Доступна ли CUDA?:", torch.cuda.is_available())
    return DataLoader, datasets, torch, transforms


@app.cell
def _(datasets, transforms):
    train_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.ToTensor()
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(root="Lab6/dataset/train", transform=train_transforms)
    test_dataset = datasets.ImageFolder(root="Lab6/dataset/test", transform=test_transforms)
    return test_dataset, train_dataset


@app.cell
def _(DataLoader, test_dataset, train_dataset):
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    return test_loader, train_loader


@app.cell
def _(torch, train_dataset):
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    class CV_CNN(nn.Module):
        def __init__(self, num_classes):
            super(CV_CNN, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=0)
            self.bn1 = nn.BatchNorm2d(32)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0)
            self.bn2 = nn.BatchNorm2d(64)
            self.fc1 = nn.Linear(64 * 14 * 14, 128)
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = x.view(-1, 64 * 14 * 14)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    num_classes = len(train_dataset.classes)
    print(num_classes)
    model = CV_CNN(num_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Модель отправлена на: {device}")
    return device, model, nn, optim


@app.cell
def _(model, nn, optim):
    crit = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return crit, optimizer


@app.cell
def _(crit, device, model, optimizer, test_loader, torch, train_loader):
    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)

            loss = crit(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = (correct_train / total_train) * 100

        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = crit(outputs, labels)

                test_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        epoch_test_loss = test_loss / len(test_loader.dataset)
        epoch_test_acc = (correct_test / total_test) * 100

        print(f"Эпоха [{epoch+1}/{epochs}] | "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}% | "
              f"Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.2f}%")
    return


@app.cell
def _(model, torch):
    torch.save(model.state_dict(), 'cv_cnn_model_weights.pth')
    print("Веса модели успешно сохранены!")
    return


@app.cell
def _(train_dataset):
    train_dataset.classes
    return


if __name__ == "__main__":
    app.run()
