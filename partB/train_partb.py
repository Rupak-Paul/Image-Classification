import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchmetrics
import pytorch_lightning as pl
import re
import torchvision.models as models
import argparse


# This class represents a pretrained model
class FineTunedPretrainModel(pl.LightningModule):
    def __init__(self, modelName='GoogLeNet', unfreezedLayersFromEnd=1):
        super().__init__()
        self.save_hyperparameters()
        
        # Defining the underlying pre-trained model
        self.backbone = None
        if modelName == 'GoogLeNet':
            self.backbone = models.googlenet(weights="DEFAULT")
        elif modelName == 'InceptionV3':
            self.backbone = models.inception_v3(weights="DEFAULT")
        elif modelName == 'ResNet50':
            self.backbone = models.resnet50(weights="DEFAULT")
        else:
            raise ValueError(f"Model: '{modelName}' is not supported.")
        
        # First, freezing all the layers
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreezing last K layers
        unfreeze_start_index = len(list(self.backbone.children())) - unfreezedLayersFromEnd
        for param in list(self.backbone.parameters())[unfreeze_start_index:]:
            param.requires_grad = True
     
        # Taking all the layers except output layer
        layers = list(self.backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.feature_extractor.eval()

        # Adding a new output layer havin 10 neurons
        num_filters = self.backbone.fc.in_features
        num_target_classes = 10
        self.classifier = nn.Linear(num_filters, num_target_classes)
        
        # Utility function to calculate accuracy
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)
        self.valid_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)
        
    def forward(self, x):
        # Forward pass through unfozen layers
        representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x

    def configure_optimizers(self):
        # Using adam optimizer to train unfrozen layers
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
    def training_step(self, batch, batch_idx):
        # Calculating and logging traning loss at each epochs
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        
        self.train_acc(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        # Calculating and logging traning accuracy at each epochs
        self.log('train_acc', self.train_acc.compute(), prog_bar=True, logger=True)
        self.train_acc.reset()
    
    def validation_step(self, batch, batch_idx):
        # Calculating and logging validation loss at each epochs
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        
        self.valid_acc(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)        
        return loss
    
    def on_validation_epoch_end(self):
        # Calculating and logging validation accuracy at each epochs
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
    model = FineTunedPretrainModel(preTrainedModelName, unfreezedLayersFromEnd)
    
    # Creating dataloader
    train_loader = DataLoader(train_set, batch_size=batchSize, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batchSize)
    
    # Training the model
    trainer = pl.Trainer(max_epochs=epochs)
    trainer.fit(model, train_loader, val_loader)

    # Calculating test accuracy
    print('Please wait, Calculating test accuracy!!')
    testAccuracy = calculateTestAccuracy(model)
    print('Test Accuracy: ', testAccuracy)
    
    # Storing the model for future use
    print('Please wait, Storing the model as a file!!')
    trainer.save_checkpoint('model.ckpt')


###############################################################################################################################

# Parsing command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train_dataset_path", "-ptrn", type=str, default="inaturalist_12K/train")
parser.add_argument("--test_dataset_path", "-ptst", type=str, default="inaturalist_12K/val")
parser.add_argument("--epochs", "-ep", type=int, default=15)
parser.add_argument("--batch_size", "-bs", type=int, default=32)
parser.add_argument("--unfreezed_layers_from_end", "-ul", type=int, default=1)
args = parser.parse_args()

# Setting dataset path form command line arguments
transformedImageSize = 224
trainDatasetPath = args.train_dataset_path
testDatasetPath = args.test_dataset_path

# Setting hyperparameters of the model from command line arguments
preTrainedModelName = 'GoogLeNet'
epochs = args.epochs
batchSize = args.batch_size
unfreezedLayersFromEnd = args.unfreezed_layers_from_end

main()