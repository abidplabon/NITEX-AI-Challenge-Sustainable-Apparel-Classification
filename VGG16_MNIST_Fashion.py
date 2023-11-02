#!/usr/bin/env python
# coding: utf-8

# In[ ]:



# In[ ]:


import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchinfo import summary

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

import random
import pandas as pd
from pathlib import Path
from tqdm.auto import trange,tqdm
import matplotlib.pyplot as plt
from timeit import default_timer as timer

# In[ ]:


train_data = datasets.FashionMNIST(root='data',
                                  train=True,
                                  transform=ToTensor(),
                                  download=True)
test_data = datasets.FashionMNIST(root='data',
                                train=False,
                                transform=ToTensor(),
                                download=True)

# In[ ]:


len(train_data), len(test_data)

# In[ ]:


class_names = train_data.classes
class_names

# In[ ]:


class_to_idx = train_data.class_to_idx
class_to_idx

# In[ ]:


img, label = train_data[0]
img.shape, label

# In[ ]:


fig = plt.figure(figsize=(12, 12))
rows, cols = 4, 4
for i in range(1, rows * cols + 1):
    img, label = train_data[i]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.permute(1,2,0))
    plt.title(class_names[label])
    plt.axis("off")

# In[ ]:


BATCH_SIZE = 32

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

len(train_loader), len(test_loader)

# In[ ]:


train_features_batch, train_labels_batch = next(iter(train_loader))
train_features_batch.shape, train_labels_batch.shape

# In[ ]:


rand_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
img, label = train_features_batch[rand_idx], train_labels_batch[rand_idx]
plt.imshow(img.permute(1,2,0))
plt.title(class_names[label])
plt.axis("off")
print(f"Image shape: {img.shape}")
print(f"Label: {class_names[label]}")

# In[ ]:


class VGG16(nn.Module):
    """
        Implementation of VGG16 architecture.

        Args:
            num_classes (int): Specify number of classes for multi-class classification task.

        Returns:
            Training loss, Training accuracy, Testing loss, Testing accuracy, Total training time.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

# In[ ]:


vgg16 = VGG16(num_classes=100)
summary(vgg16, input_size=(32,3,224,224),col_names=['input_size','output_size','num_params','trainable'],col_width=25)

# In[ ]:


class tinyVGG(nn.Module):
    """Implementation of tinyVGG model.

        Args:
            input_shape - Input tensor shape.
            hidden_units - Number of units for the intermediate convolution layers.
            output_shape - Output tensor shape.
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7,
                      out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

# In[ ]:


tinyvgg = tinyVGG(input_shape=1,
                 hidden_units=32,
                 output_shape=len(class_names))
summary(tinyvgg, input_size=(32,1,28,28),col_names=['input_size','output_size','num_params','trainable'],col_width=25)

# In[ ]:


def print_train_time(start, end, device=None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format).
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time

# In[ ]:


def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

# In[ ]:


device = "cuda" if torch.cuda.is_available() else "cpu"

# In[ ]:


def train_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    accuracy_fn,
    device: torch.device = device):
    """Performs a single step of training.

        Args:
            model: PyTorch model to train.
            data_loader: DataLoader object to load train/ test image data.
            loss_fn: Loss function to train model on.
            optimizer: Optimizer to update model weights and biases.
            accuracy_fn: Calculates model accuracy, either train/test.
            device: Device (CPU/GPU/TPU)

        Returns:
            Training loss, Training accuracy.
    """
    # Initialize training loss and accuracy
    train_loss, train_acc = 0, 0

    # Set model to 'train' mode.
    model.train()

    # Iterate through DataLoader
    for batch, (X, y) in enumerate(data_loader):

        # Send data to device
        X, y = X.to(device), y.to(device)

        # Get predictions from model
        y_pred = model(X)

        # Compute model loss and accuracy
        loss = loss_fn(y_pred, y)
        acc = accuracy_fn(y, y_pred.argmax(dim=1))

        # Accumulate training loss and accuracy
        train_loss += loss
        train_acc += acc

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Compute average training loss and accuracy
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    print(f'Train loss: {train_loss:.4f} | Train accuracy: {train_acc:.4f}')

# In[ ]:


def test_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    accuracy_fn,
    device: torch.device = device):
    """Performs a single step of testing.

        Args:
            model: PyTorch model to test.
            data_loader: DataLoader object to load train/ test image data.
            loss_fn: Loss function to test model on.
            accuracy_fn: Calculates model accuracy, either train/test.
            device: Device (CPU/GPU/TPU)

        Returns:
            Testing loss, Testing accuracy.
    """
    # Initialize testing loss and accuracy
    test_loss, test_acc = 0, 0

    # Set model to 'evaluation' mode
    model.eval()

    # Using torch.inference_mode() to ensure zero gradients, compute testing loss and accuracy
    with torch.inference_mode():
        # Iterate through DataLoader
        for batch, (X, y) in enumerate(data_loader):

            # Send data to device
            X, y = X.to(device), y.to(device)

            # Get predictions from model
            y_pred = model(X)

            # Compute model loss and accuracy
            loss = loss_fn(y_pred, y)
            acc = accuracy_fn(y, y_pred.argmax(dim=1))

            # Accumulate testing loss and accuracy
            test_loss += loss
            test_acc += acc

        # Compute average testing loss and accuracy
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)

    print(f'Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}')


# In[ ]:


def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    accuracy_fn,
    epochs: int = 10,
    device: torch.device = device):
    """Trains a model for specified number of epochs.

        Args:
            model: PyTorch model to train.
            train_loader: DataLoader object to load training image data.
            test_loader: DataLoader object to load testing image data.
            loss_fn: Loss function to train model on.
            optimizer: Optimizer to update model weights and biases.
            accuracy_fn: Calculates model accuracy, either train/test.
            epochs: Number of epochs to train model for.
            device: Device (CPU/GPU/TPU)

        Returns:
            Training loss, Training accuracy, Testing loss, Testing accuracy, Total Training time.
    """
    train_time_start = timer()
    for epoch in trange(epochs):
        print(f'Epoch: {epoch} \n -----------------------')

        train_step(model, train_loader, loss_fn, optimizer, accuracy_fn, device)
        test_step(model, test_loader, loss_fn, accuracy_fn, device)

    train_time_end = timer()
    print_train_time(train_time_start, train_time_end, device=device)

# In[ ]:


epochs = 7
learning_rate = 3e-4  # Karpathy constant
num_classes = 100

model = tinyVGG(input_shape=1,
               hidden_units=32,
               output_shape=len(class_names)).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
train_model(model,
           train_loader,
           test_loader,
           loss_fn,
           optimizer,
           accuracy_fn,
           epochs,
           device)

# In[ ]:


def eval_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    accuracy_fn,
    device: torch.device=device):
    """Returns a dictionary containing the results of model predictions on data_loader.

       Args:
           model: PyTorch model to evaluate.
           data_loader: DataLoader object to load testing images.
           loss_fn: Loss function to test model on.
           accuracy_fn: Calculates model accuracy, either train/test.
           device: Device (CPU/GPU/TPU)
    """
    # Initialize loss and accuracy
    loss, acc = 0, 0

    # Set model to 'evaluate' mode
    model.eval()

    # Compute loss and accuracy with torch.inference_mode() to ensure zero gradients
    with torch.inference_mode():
        # Iterate through data_loader
        for X, y in tqdm(data_loader):
            # Send data to device
            X, y = X.to(device), y.to(device)

            # Computer model predictions
            y_pred = model(X)

            # Compute and accumulate loss and accuracy
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y, y_pred.argmax(dim=1))

        # Compute average loss and accuracy
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {
        'Model name' : model.__class__.__name__,
        'Model loss' : loss.item(),
        'Model accuracy' : acc
    }


# In[ ]:


tinyVGG_results = eval_model(model, test_loader, loss_fn, accuracy_fn)
tinyVGG_results

# In[ ]:


def make_predictions(
    model: torch.nn.Module,
    data: list,
    device: torch.device=device):

    """Returns model's prediction probability tensor.

       Args:
           model: PyTorch model to make predictions.
           data: Data on which model will make predictions.
           device: Device (CPU/GPU/TPU)
    """
    # Initialize prediction probability list
    pred_probs = []

    # Set model to 'evaluate' mode
    model.eval()
  # Compute model prediction probabilities in torch.inference_mode() to ensure zero gradients
    with torch.inference_mode():
        # Iterate through data to generate predictions
        for sample in data:

            # Reshaping data and sending it to device
            sample = torch.unsqueeze(sample, dim=0).to(device)

            # Compute prediction logit from model
            pred_logit = model(sample)

            # Compute prediction probability from prediction logit by apply softmax function
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)

            # Extending list of prediction probabilities
            pred_probs.append(pred_prob.cpu())
    return torch.stack(pred_probs)

# In[ ]:


test_samples = []
test_labels = []

for sample, label in random.sample(list(test_data),k=9):
    test_samples.append(sample)
    test_labels.append(label)

pred_probs = make_predictions(model, test_samples)
pred_classes = pred_probs.argmax(dim=1)
pred_probs, pred_classes

# In[ ]:


plt.figure(figsize=(12,12))
nrows = 3
ncols = 3
for i, sample in enumerate(test_samples):
    plt.subplot(nrows, ncols, i+1)
    plt.imshow(sample.squeeze())
    title_txt = f"True: {class_names[test_labels[i]]} | Pred: {class_names[pred_classes[i]]}"
    plt.axis("off")
    if class_names[pred_classes[i]] == class_names[test_labels[i]]:
        plt.title(title_txt,c='g')
    else:
        plt.title(title_txt,c='r')

# In[ ]:


y_preds = []
model.eval()
with torch.inference_mode():
    for X, y in tqdm(test_loader, 'Making predictions...'):
        X, y = X.to(device), y.to(device)
        y_logit = model(X)
        y_pred = torch.softmax(y_logit.squeeze(), dim=0).argmax(dim=1)

        y_preds.append(y_pred.cpu())

y_pred_tensor = torch.cat(y_preds)
y_pred_tensor

# In[ ]:


from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

confmat = ConfusionMatrix(task='multiclass',num_classes=len(class_names))
confmat_tensor = confmat(preds=y_pred_tensor, target=test_data.targets)

fig, ax = plot_confusion_matrix(conf_mat=confmat_tensor.numpy(),
                                class_names=class_names,
                                figsize=(10,7))

# In[ ]:




# In[ ]:


#cd drive/MyDrive/NITEX AI Challenge Sustainable Apparel Classification

# In[ ]:


#cd ..

# In[ ]:


MODEL_PATH = Path('models')
MODEL_PATH.mkdir(parents=True,exist_ok=True)

MODEL_NAME = 'tinyVGG_fashionMNIST.pth'
MODEL_SAVE_PATH = MODEL_PATH/MODEL_NAME

print(f'Saving model to: {MODEL_SAVE_PATH}')
torch.save(model.state_dict(), MODEL_SAVE_PATH)

# In[ ]:


import h5py

# Assuming you have a PyTorch model named 'model' that you want to save

MODEL_PATH = Path('models')
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = 'fashionMNIST.h5'
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(f'Saving model to: {MODEL_SAVE_PATH}')

# Open an HDF5 file in write mode
with h5py.File(MODEL_SAVE_PATH, 'w') as hf:
    # Create a group to store the model's parameters
    model_group = hf.create_group('model_parameters')

    # Iterate through the model's parameters and save them
    for name, param in model.state_dict().items():
        model_group.create_dataset(name, data=param.cpu().numpy())

#Save any additional metadata or information you may need
#For example, you can save the model's architecture, training history, etc.

