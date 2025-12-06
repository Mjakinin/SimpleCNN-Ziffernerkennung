# SimpleCNN – Ziffernerkennung mit PyTorch (MNIST)

Dieses Projekt ist ein kleines End-to-End Deep-Learning-Projekt in Python / PyTorch.

Ein Convolutional Neural Network (CNN) wird auf dem MNIST-Datensatz trainiert, um
handgeschriebene Ziffern (0–9) zu erkennen. Zusätzlich können eigene Bilder
(z.B. in Paint gezeichnete Ziffern) geladen und klassifiziert werden.

## Features

- Laden und Vorverarbeitung des MNIST-Datensatzes
- Convolutional Neural Network mit:
  - Conv2d + ReLU + MaxPool2d Schichten
  - Fully Connected Layern für die Klassifikation
- Training auf GPU (CUDA), falls verfügbar
- Auswertung auf dem Testset (≈ 99 % Accuracy)
- Visualisierung von:
  - Trainings-Loss pro Epoche
  - Test-Accuracy pro Epoche
  - Beispiel-MNIST-Bildern mit vorhergesagter Ziffer
- Klassifikation **eigener Bilder** (PNG), inkl. Anzeige mit Vorhersage

## Installation

```bash
# Im Projektordner
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
# SimpleCNN-Ziffernerkennung
