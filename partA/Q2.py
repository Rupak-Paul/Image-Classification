import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb


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



def main():
    # Defining wandb sweep configuration
    sweep_config = {
        'method': 'bayes',
        'name' : 'sweep cross entropy',
        'metric': {
            'name': 'val_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'epochs': {'values': [10, 15]},
            'optimizer': {'values':['sgd', 'adam']},
            'activationFunctions': {'values':['ReLU', 'GELU', 'SiLU', 'Mish']},
            'batchSize': {'values':[8, 16, 32, 64]},
            'batchNormalization': {'values':['no', 'yes']},
            'noOfFilterInFirstLayer': {'values':[16, 32, 64, 128]},
            'filterOrganization': {'values':['equal', 'half']},
            'convKernalSize': {'values':[3, 5, 7]},
            'convStrideSize': {'values':[1, 2]},
            'maxpoolKernalSize': {'values':[2, 3]},
            'maxpoolStrideSize': {'values':[1, 2]},
            'denseLayerSize': {'values':[64, 128, 256, 512]}
        }
    }
    
    # Executing the sweep for 40 runs
    wandb.login(key='64b2775be5c91a3a2ab0bac3d540a1d9f6ea7579')
    sweep_id = wandb.sweep(sweep=sweep_config, project='CS23M056_DL_Assignment_2')
    wandb.agent(sweep_id, function=callMe, count=40)
    wandb.finish()


def callMe():
    wandb.init(project='CS23M056_DL_Assignment_2')
    
    # Hyperparameters of the model
    epochs = wandb.config['epochs']
    optimizer = wandb.config['optimizer']
    activationFunction = wandb.config['activationFunctions']
    batchSize = wandb.config['batchSize']
    batchNormalization = wandb.config['batchNormalization']
    noOfFilterInFirstLayer = wandb.config['noOfFilterInFirstLayer']
    filterOrganization = wandb.config['filterOrganization']
    convKernalSize = wandb.config['convKernalSize']
    convStrideSize = wandb.config['convStrideSize']
    maxpoolKernalSize = wandb.config['maxpoolKernalSize']
    maxpoolStrideSize = wandb.config['maxpoolStrideSize']
    denseLayerSize = wandb.config['denseLayerSize']
    
    # Setting wandb run name
    wandb.run.name = 'ep-'+str(epochs) + '-op-'+str(optimizer) + '-act-'+str(activationFunction) + '-bs-'+str(batchSize) + '-bn-'+str(batchNormalization) + '-nof-'+str(noOfFilterInFirstLayer) + '-fog-'+str(filterOrganization) + '-cks-'+str(convKernalSize) + '-css-'+str(convStrideSize) + '-mks-'+str(maxpoolKernalSize) + '-mss-'+str(maxpoolStrideSize) + '-dls-'+str(denseLayerSize)
    
    # Creating the model
    model = SmallCNN(transformedImageSize, optimizer, activationFunction, batchNormalization, noOfFilterInFirstLayer, filterOrganization, convKernalSize, convStrideSize, maxpoolKernalSize, maxpoolStrideSize, denseLayerSize)
    
    # Loading dataset
    train_loader = DataLoader(train_set, batch_size=batchSize, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batchSize)
    
    # Traning the model
    trainer = pl.Trainer(max_epochs=epochs, devices=-1, logger=wandb_logger)
    trainer.fit(model, train_loader, val_loader)

###############################################################################################################################

# Resizing the input image to 224x224
transformedImageSize = 224

# Define data transforms
transform = transforms.Compose([
    transforms.Resize((transformedImageSize, transformedImageSize)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset using ImageFolder or any other dataset loader
dataset = ImageFolder(root='/kaggle/input/inaturalist-12k/inaturalist_12K/train', transform=transform)

# Split dataset into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

# Creating wandb logger
wandb_logger = WandbLogger(project="CS23M056_DL_Assignment_2", log_model='all')

main()