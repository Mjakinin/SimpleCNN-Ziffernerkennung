import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image  # NEU: für eigene Bilder


def get_device():
    if torch.cuda.is_available():
        print("✅ GPU gefunden, benutze CUDA")
        return torch.device("cuda")
    else:
        print("⚠️ Keine GPU gefunden, benutze CPU")
        return torch.device("cpu")


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Eingabe: 1x28x28 (MNIST)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 16x28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                            # 16x14x14

            nn.Conv2d(16, 32, kernel_size=3, padding=1),# 32x14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                            # 32x7x7
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # 10 Klassen (Ziffern 0–9)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # [Batch, 32*7*7]
        x = self.fc_layers(x)
        return x


def get_dataloaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root="data",
        train=True,
        transform=transform,
        download=True
    )

    test_dataset = datasets.MNIST(
        root="data",
        train=False,
        transform=transform,
        download=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    return avg_loss


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    return acc


def show_example_predictions(model, loader, device):
    """Zeigt ein paar MNIST-Testbilder mit Vorhersage."""
    model.eval()
    images, labels = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        preds = outputs.argmax(dim=1)

    fig, axes = plt.subplots(2, 3, figsize=(8, 5))
    axes = axes.flatten()

    for img, label, pred, ax in zip(images[:6], labels[:6], preds[:6], axes):
        img_cpu = img.cpu().squeeze(0)  # [1,28,28] -> [28,28]
        ax.imshow(img_cpu, cmap="gray")
        ax.set_title(f"True: {label.item()} | Pred: {pred.item()}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def predict_custom_image(model, device, image_path):
    """
    Lädt ein eigenes Bild (z.B. meine3.png),
    bereitet es so vor wie MNIST und zeigt es mit Vorhersage an.
    """
    # Gleiche Normalisierung wie bei MNIST
    transform = transforms.Compose([
        transforms.Grayscale(),          # sicherstellen: 1 Kanal
        transforms.Resize((28, 28)),     # auf 28x28 bringen
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Bild laden
    img = Image.open(image_path)
    img_converted = transform(img).unsqueeze(0).to(device)  # [1,1,28,28]

    model.eval()
    with torch.no_grad():
        output = model(img_converted)
        pred = output.argmax(dim=1).item()

    # Originalbild anzeigen (nicht das normalisierte)
    plt.figure()
    plt.imshow(img.convert("L"), cmap="gray")
    plt.title(f"Datei: {image_path}\nVorhersage: {pred}")
    plt.axis("off")
    plt.show()

    print(f"➡️ Vorhersage für {image_path}: {pred}")
    return pred


def main():
    device = get_device()

    train_loader, test_loader = get_dataloaders(batch_size=64)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 5
    train_losses = []
    test_accuracies = []

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_acc = evaluate(model, test_loader, device)

        train_losses.append(train_loss)
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f} | Test Acc: {test_acc*100:.2f}%")

    torch.save(model.state_dict(), "simple_cnn_mnist.pth")
    print("✅ Training fertig, Modell gespeichert als simple_cnn_mnist.pth")

    # Loss-Verlauf plotten
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.show()

    # Accuracy-Verlauf plotten
    plt.figure()
    plt.plot(range(1, num_epochs + 1), [acc * 100 for acc in test_accuracies], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy [%]")
    plt.title("Test Accuracy")
    plt.grid(True)
    plt.show()

    # Beispiel-MNIST-Bilder mit Vorhersage
    show_example_predictions(model, test_loader, device)

    custom_images = [
        "custom_images/meine3.png",
        "custom_images/meine4.png",
        "custom_images/meine8.png",
        "custom_images/meine9.png"
    ]

    for img_path in custom_images:
        try:
            predict_custom_image(model, device, img_path)
        except FileNotFoundError:
            print(f"⚠️ Datei nicht gefunden: {img_path} (Pfad prüfen!)")


if __name__ == "__main__":
    main()
