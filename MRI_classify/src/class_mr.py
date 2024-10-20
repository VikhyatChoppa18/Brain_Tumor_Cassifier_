import numpy as np
import pandas as pd
import os
import plotly.express as px
import pickle
import zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
from glob import glob
from fastapi import FastAPI, File, UploadFile
from io import BytesIO

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BrMRIDataset(Dataset):
    def __init__(self, data_dir, reshape=True, width=130, height=130):
        self.data_dir = data_dir
        self.healthy = glob(os.path.join(data_dir, 'Brain Tumor Data Set','Brain Tumor Data Set','Healthy', '*'))
        self.tumor = glob(os.path.join(data_dir, 'Brain Tumor Data Set','Brain Tumor Data Set','Brain Tumor', '*'))

        self.height = height
        self.width = width
        self.reshape = reshape

        image_labels = [0] * len(self.healthy) + [1] * len(self.tumor)
        images = self.healthy + self.tumor

        self.dataframe = pd.DataFrame({"image": images, "label": image_labels})
        self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe['image'][idx]
        label = self.dataframe['label'][idx]

        image = Image.open(img_path).convert("L")
        if self.reshape:
            image = image.resize((self.width, self.height))

        arr = np.array(image).reshape(1, self.width, self.height)
        return torch.tensor(arr, device=device).float(), torch.tensor(label, device=device)


class TumorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


def train_model(model, train_loader, val_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return train_losses, val_losses


def main():
    with zipfile.ZipFile('../data/archive.zip', 'r') as zip_ref:
        zip_ref.extractall('../tumor_data')

    dataset = BrMRIDataset("../tumor_data/")
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    # Initializing and train model
    model = TumorModel().to(device)
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=10)

    fig = px.line(x=range(1, 11), y=[train_losses, val_losses],
                  labels={'x': 'Epoch', 'y': 'Loss'},
                  title='Training and Validation Loss',
                  line_shape='linear')
    fig.show()

    torch.save(model.state_dict(), "../model/brain_tumor_mri_model.pth")


if __name__ == "__main__":
    main()
