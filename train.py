import torch
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np


CLASSES = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum',
           'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']

class VegetableDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []


        for idx, label in enumerate(CLASSES):
            class_dir = os.path.join(self.root_dir, label)
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.jpg'):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


BATCH_SIZE = 16
NUM_EPOCHS = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Active device: {device}')


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.255])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.255])
    ])
}


train_dataset = VegetableDataset('vegetables/train', transform=data_transforms['train'])
val_dataset = VegetableDataset('vegetables/test', transform=data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


model = models.resnet18(pretrained=True)


for param in model.parameters():
    param.requires_grad = False


model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
model.to(device)


loss_func = torch.nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.fc.parameters(), lr=0.001)


if __name__ == '__main__':
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f'\nEpoch: {epoch} / {NUM_EPOCHS}')
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimiser.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimiser.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        print(f'Train loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')


    torch.save(model.state_dict(), 'vegetable_resnet18.pth')


    model.eval()
    with torch.no_grad():
        images_shown = 0
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for i in range(inputs.size(0)):
                inp = inputs[i].cpu()
                inp_np = inp.numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.255])
                inp_np = std * inp_np + mean
                inp_np = np.clip(inp_np, 0, 1)

                plt.imshow(inp_np)
                plt.title(f"Predicted: {CLASSES[preds[i]]} | Actual: {CLASSES[labels[i]]}")
                plt.axis('off')
                plt.show()

                images_shown += 1
                if images_shown >= 5:
                    break
            if images_shown >= 5:
                break
