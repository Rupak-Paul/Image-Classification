# CS6910-Assignment-2
This is [CS6910](http://www.cse.iitm.ac.in/~miteshk/CS6910.html) course assignment-2 at IIT Madras. [Here](https://wandb.ai/cs6910_2024_mk/A1/reports/CS6910-Assignment-2--Vmlldzo3MjcwNzM1) you will find detailed information about the assignment. This assignment is divided into two parts. In **partA** I created a CNN and trained it on [iNaturalist dataset](https://storage.googleapis.com/wandb_datasets/nature_12K.zip). In **partB** I used a pre-trained model (GoogLeNet) and fine-tuned it for [iNaturalist dataset](https://storage.googleapis.com/wandb_datasets/nature_12K.zip).

[Here](https://wandb.ai/cs23m056/CS23M056_DL_Assignment_2/reports/CS6910-Assignment-2--Vmlldzo3NDQ2OTM0) is the detailed wandb report for this assignment.

# Part-A
In partA I used these libraies
```
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchmetrics
import pytorch_lightning as pl
import argparse
```

If you don't have these packages installed then use this command to install 
```
pip install pytorch
pip install torchvision
pip install pytorch_lightning
```

If you want to **train** and **use** this model for your **dataset** then first download the **train_parta.py** file. Hyperparameters, train and test dataset path can be mentioned using command line arguments. Here are the supported command line arguments:

| Command | Description | Accepted Values | Default Value |
| ------- | ----------- | --------------- | ------------- |
| --train_dataset_path, -ptrn | Path to the training dataset | String | inaturalist_12K/train |
| --test_dataset_path, -ptst | Path to the testing dataset | String | inaturalist_12K/val |
| --epochs, -ep | Number of epochs for training | Integer | 15 |
| --optimizer, -opt | Optimizer for training | 'sgd', 'adam' | adam |
| --activation, -act | Activation function for the model | 'ReLU', 'GELU', 'SiLU', 'Mish' | ReLU |
| --batch_size, -bs | Batch size for training | Integer | 32 |
| --batch_normalization, -bn | Whether to use batch normalization | 'no', 'yes' | yes |
| --filter_in_first_layer, -nf | Number of filters in the first convolutional layer | Integer | 128 |
| --filter_organization, -fo | Organization of filters in convolutional layers | 'equal', 'half' | equal |
| --conv_kernel_size, -cks | Size of the convolutional kernel | Integer | 3 |
| --conv_stride_size, -css | Stride size for convolutional layers | Integer | 2 |
| --maxpool_kernel_size, -mks | Size of the maxpooling kernel | Integer | 3 |
| --maxpool_stride_size, -mss | Stride size for maxpooling layers | Integer | 1 |
| --dense_layer_size, -dls | Number of neurons in the dense layer | Integer | 64 |

# Part-B
In partB I used these libraies
```
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchmetrics
import pytorch_lightning as pl
import torchvision.models as models
import argparse
```
If you don't have these packages installed then use this command to install 
```
pip install pytorch
pip install torchvision
pip install pytorch_lightning
```
If you want to **train** and **use** this model for your **dataset** then first download the **train_partb.py** file. Hyperparameters, train and test dataset path can be mentioned using command line arguments. Here are the supported command line arguments:
| Command | Description | Accepted Values | Default Value |
| ------- | ----------- | --------------- | ------------- |
| --train_dataset_path, -ptrn | Path to the training dataset | String | inaturalist_12K/train |
| --test_dataset_path, -ptst | Path to the testing dataset | String | inaturalist_12K/val |
| --epochs, -ep | Number of epochs for training | Integer | 15 |
| --batch_size, -bs | Batch size for training | Integer | 256 |
| --unfreezed_layers_from_end, -ul | Number of unfreezed layers from the end | Integer | 0 |
