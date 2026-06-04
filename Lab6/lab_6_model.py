import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    import torch.nn.functional as F
    from PIL import Image

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

    return CV_CNN, torch


@app.cell
def _(CV_CNN):
    num_classes = 20
    model_inference = CV_CNN(num_classes)
    return (model_inference,)


@app.cell
def _(model_inference, torch):
    weights_path = "cv_cnn_model_weights.pth" 
    model_inference.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model_inference.eval()
    return


@app.cell
def _(mo):
    file_input = mo.ui.file(
        filetypes=["image/jpeg", "image/png"],
        kind="area",
        label="Загрузите фото вашей руки (JPEG/PNG)"
    )
    file_input
    return (file_input,)


@app.cell
def _(file_input, mo, model_inference, torch):
    def _():
        if file_input.value is None or len(file_input.value) == 0:
            return mo.md("*Ожидание загрузки изображения руки...*")

        import io
        from PIL import Image
        import torchvision.transforms as transforms

        image_bytes = file_input.value[0].contents
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        inference_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.CenterCrop(min(img.size)),
            transforms.Resize((64, 64)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
        ])
        img_tensor = inference_transforms(img)
        mask = (img_tensor[0:1, :, :] > 0.3).float()
        img_tensor = mask.repeat(3, 1, 1)
        img_tensor = img_tensor.unsqueeze(0)
        my_classes = [
          "0",
          "1",
          "10",
          "11",
          "12",
          "13",
          "14",
          "15",
          "16",
          "17",
          "18",
          "19",
          "2",
          "3",
          "4",
          "5",
          "6",
          "7",
          "8",
          "9"
        ]
        #my_classes = [f"{i}" for i in range(num_classes)] 

        with torch.no_grad():
            outputs = model_inference(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted_idx = torch.max(outputs, 1)

        class_idx = predicted_idx.item()
        confidence = probabilities[0][class_idx].item() * 100
        result_text = my_classes[class_idx]
        processed_img = transforms.ToPILImage()(img_tensor[0])
        debug_mask = transforms.ToPILImage()(img_tensor[0])

        return mo.vstack([
            mo.md(f"""
            ### Результат распознавания:
            * **Предсказанный жест:** `{result_text}`
            * **Уверенность сети:** `{confidence:.2f}%`
            """),
            mo.hstack([
                mo.vstack([mo.md("**Оригинал:**"), mo.image(src=img, width=200)]),
                mo.vstack([mo.md("**Что пришло в модель:**"), mo.image(src=debug_mask, width=200)])
            ], gap=2)
        ])

    _()
    return


if __name__ == "__main__":
    app.run()
