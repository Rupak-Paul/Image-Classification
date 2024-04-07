import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchmetrics
import pytorch_lightning as pl

# This class represents a Convolutional Neural Network
class SmallCNN(pl.LightningModule):
    def __init__(self, imageSize, optimizer='sgd', activation='ReLU', batch_norm='no', noOfFilterInFirstLevel=16, filterOrganization='equal', convKernelSize=3, convStrideSize=1, maxPoolKernalSize=2, maxPoolStrideSize=1, denseLayerSize=128):
        super(SmallCNN, self).__init__()
        
        # Setting the optimizer
        self.optimizer = optimizer
        
        # Calculating no of filters in each convolution layer based on no of filter in first layer and filter organization
        noOfFiltersInConvLayers = []
        if filterOrganization == 'equal':
            noOfFiltersInConvLayers = [noOfFilterInFirstLevel, noOfFilterInFirstLevel, noOfFilterInFirstLevel, noOfFilterInFirstLevel, noOfFilterInFirstLevel]
        elif filterOrganization == 'double':
            noOfFiltersInConvLayers = [noOfFilterInFirstLevel, 2*noOfFilterInFirstLevel, 4*noOfFilterInFirstLevel, 8*noOfFilterInFirstLevel, 16*noOfFilterInFirstLevel]
        elif filterOrganization == 'half':
            noOfFiltersInConvLayers = [noOfFilterInFirstLevel, noOfFilterInFirstLevel//2, noOfFilterInFirstLevel//4, noOfFilterInFirstLevel//8, noOfFilterInFirstLevel//16]
        else:
            raise ValueError(f"filter organization '{filterOrganization}' is not supported.")
        
        # Creating 5 convolution layer
        self.conv1 = nn.Conv2d(3, noOfFiltersInConvLayers[0], kernel_size=convKernelSize, stride=convStrideSize)
        self.conv2 = nn.Conv2d(noOfFiltersInConvLayers[0], noOfFiltersInConvLayers[1], kernel_size=convKernelSize, stride=convStrideSize)
        self.conv3 = nn.Conv2d(noOfFiltersInConvLayers[1], noOfFiltersInConvLayers[2], kernel_size=convKernelSize, stride=convStrideSize)
        self.conv4 = nn.Conv2d(noOfFiltersInConvLayers[2], noOfFiltersInConvLayers[3], kernel_size=convKernelSize, stride=convStrideSize)
        self.conv5 = nn.Conv2d(noOfFiltersInConvLayers[3], noOfFiltersInConvLayers[4], kernel_size=convKernelSize, stride=convStrideSize)
        
        # Creating batch normalization layer
        if batch_norm == 'yes':
            self.bn1 = nn.BatchNorm2d(noOfFiltersInConvLayers[0])
            self.bn2 = nn.BatchNorm2d(noOfFiltersInConvLayers[1])
            self.bn3 = nn.BatchNorm2d(noOfFiltersInConvLayers[2])
            self.bn4 = nn.BatchNorm2d(noOfFiltersInConvLayers[3])
            self.bn5 = nn.BatchNorm2d(noOfFiltersInConvLayers[4])
        
        # Creating maxpool layer. I am using same maxpool layer at every levels
        self.pool = nn.MaxPool2d(kernel_size=maxPoolKernalSize, stride=maxPoolStrideSize)
        
        # Computing input size of the fully connected layer
        fc_input_size_temp = (imageSize - convKernelSize) // convStrideSize + 1
        fc_input_size_temp = (fc_input_size_temp - maxPoolKernalSize) // maxPoolStrideSize + 1
        for i in range(4):
            fc_input_size_temp = (fc_input_size_temp - convKernelSize) // convStrideSize + 1
            fc_input_size_temp = (fc_input_size_temp - maxPoolKernalSize) // maxPoolStrideSize + 1
        
        self.fc_input_size = noOfFiltersInConvLayers[4] * fc_input_size_temp * fc_input_size_temp
        
        # Creating fully connected layer
        self.fc1 = nn.Linear(self.fc_input_size, denseLayerSize)
        
        # Creating output layer
        self.fc2 = nn.Linear(denseLayerSize, 10)
        
        # Setting the activation function
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'GELU':
            self.activation = nn.GELU()
        elif activation == 'SiLU':
            self.activation = nn.SiLU()
        elif activation == 'Mish':
            self.activation = nn.Mish()
        else:
            raise ValueError(f"Activation function '{activation}' is not supported.")
        
        # Utility functions to calculate train accuracy and validation accuracy
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)
        self.valid_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)
        
    def forward(self, x):
        # Forward propagation for a input x
        x = self.activation(self.conv1(x))
        if hasattr(self, 'bn1'):
            x = self.bn1(x)
        x = self.pool(x)
        
        x = self.activation(self.conv2(x))
        if hasattr(self, 'bn2'):
            x = self.bn2(x)
        x = self.pool(x)
        
        x = self.activation(self.conv3(x))
        if hasattr(self, 'bn3'):
            x = self.bn3(x)
        x = self.pool(x)
        
        x = self.activation(self.conv4(x))
        if hasattr(self, 'bn4'):
            x = self.bn4(x)
        x = self.pool(x)
        
        x = self.activation(self.conv5(x))
        if hasattr(self, 'bn5'):
            x = self.bn5(x)
        x = self.pool(x)
        
        x = x.view(-1, self.fc_input_size)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

    def configure_optimizers(self):
        # Configuring the optimizer
        _optimizer = None
        if self.optimizer == 'sgd':
            _optimizer = torch.optim.SGD(self.parameters(), lr=0.001)
        elif self.optimizer == 'adam':
            _optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        else:
            raise ValueError(f"Optimizer '{self.optimizer}' is not supported.")
        return _optimizer

    def training_step(self, batch, batch_idx):
        # Calculating and logging traning loss at the end of each epochs
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        
        preds = torch.argmax(y_hat, dim=1)
        self.train_acc(preds, y)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        # Calculating and logging tranig accuracy at the end of each epochs
        self.log('train_acc', self.train_acc.compute(), prog_bar=True, logger=True)
        self.train_acc.reset()
    
    def validation_step(self, batch, batch_idx):
        # Calculating and logging validation loss at the end of each epochs
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        
        preds = torch.argmax(y_hat, dim=1)
        self.valid_acc(preds, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)        
        return loss
    
    def on_validation_epoch_end(self):
        # Calculating and logging validation accuracy at the end of each epochs
        self.log('val_accuracy', self.valid_acc.compute(), prog_bar=True, logger=True)
        self.valid_acc.reset()


def calculateTestAccuracy(model):
    # Define data transforms
    transform = transforms.Compose([
        transforms.Resize((transformedImageSize, transformedImageSize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Loading test dataset
    testDataset = ImageFolder(root=testDatasetPath, transform=transform)
    test_dataloader = DataLoader(testDataset, shuffle=True)
    
    # Calculating test accuracy
    corrctPrediction = 0
    for i, (image, label) in enumerate(test_dataloader):
        with torch.no_grad():
            output = model(image)
        
        _, predicted = torch.max(output, 1)

        if(predicted == label):
            corrctPrediction += 1
    
    return (corrctPrediction / len(test_dataloader))


def main():    
    # Define data transforms
    transform = transforms.Compose([
        transforms.Resize((transformedImageSize, transformedImageSize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset using ImageFolder or any other dataset loader
    testDataset = ImageFolder(root=testDatasetPath, transform=transform)

    # Split dataset into train and validation sets
    train_size = int(0.8 * len(testDataset))
    val_size = len(testDataset) - train_size
    train_set, val_set = torch.utils.data.random_split(testDataset, [train_size, val_size])
    
    # Creating the model
    model = SmallCNN(transformedImageSize, optimizer, activationFunction, batchNormalization, noOfFilterInFirstLayer, filterOrganization, convKernalSize, convStrideSize, maxpoolKernalSize, maxpoolStrideSize, denseLayerSize)
    
    # Creating dataloader
    train_loader = DataLoader(train_set, batch_size=batchSize, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batchSize)
    
    # Training the model
    trainer = pl.Trainer(max_epochs=epochs, devices=-1)
    trainer.fit(model, train_loader, val_loader)

    # Calculating test accuracy
    print('Please wait, Calculating test accuracy!!')
    testAccuracy = calculateTestAccuracy(model)
    print('Test Accuracy: ', testAccuracy)
    
    # Storing the model for future use
    print('Please wait, Storing the model as a file!!')
    trainer.save_checkpoint('model.ckpt')
    
###############################################################################################################################

# Dataset path
transformedImageSize = 224
trainDatasetPath = 'inaturalist_12K/train'
testDatasetPath = 'inaturalist_12K/val'

# Hyperparameters of the model
epochs = 15
optimizer = 'adam'
activationFunction = 'ReLU'
batchSize = 32
batchNormalization = 'yes'
noOfFilterInFirstLayer = 128
filterOrganization = 'equal'
convKernalSize = 3
convStrideSize = 2
maxpoolKernalSize = 3
maxpoolStrideSize = 1
denseLayerSize = 64

main()