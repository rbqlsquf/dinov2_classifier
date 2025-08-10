import torch
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import torchvision
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

# Load the DINOv2 model 
dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

class CustomCIFAR10Dataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_path = self.file_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label

class CustomCIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, transform):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transform

    def prepare_data(self):
        torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
        torchvision.datasets.CIFAR10(root="./data", train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, transform=self.transform)
            num_train = len(train_dataset)
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(train_dataset, [num_train - 5000, 5000])

        if stage == 'test' or stage is None:
            self.test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, transform=self.transform)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

class LinearClassifierHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.head(x)

class CustomModel(pl.LightningModule):
    def __init__(self, embed_dim, num_classes, learning_rate=0.001):
        super().__init__()
        self.dinov2_vits14 = dinov2_vits14.eval()
        self.linear_classifier_head = LinearClassifierHead(embed_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.validation_losses = []

    def forward(self, x):
        with torch.no_grad():
            features = self.dinov2_vits14(x)
        return self.linear_classifier_head(features)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.linear_classifier_head.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        val_images, val_labels = batch
        val_outputs = self(val_images)
        val_loss = self.criterion(val_outputs, val_labels)
        self.validation_losses.append(val_loss.item())
        return val_loss

    def on_validation_epoch_end(self):
        avg_val_loss = sum(self.validation_losses) / len(self.validation_losses)
        self.log('val_loss', avg_val_loss, prog_bar=True)
        self.validation_losses = []

def main():
    # Configuration
    batch_size = 1024
    num_epochs = 2
    learning_rate = 0.001
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_module = CustomCIFAR10DataModule(batch_size=batch_size, transform=transform)
    model = CustomModel(embed_dim=dinov2_vits14.embed_dim, num_classes=10, learning_rate=learning_rate)
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        check_val_every_n_epoch=1,
    )

    # 시간 측정 시작
    start_time = time.time()

    # 모델 훈련 및 검증
    trainer.fit(model, data_module)
    trainer.validate(model, datamodule=data_module)

    # 시간 측정 종료
    end_time = time.time()

    print(f"Training and testing complete. Total time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 