"""
Module to run training for image classification and evaluate trained model.
The module defines following commandline arguments:
* `--epochs`: An integer argument specifying the number of training epochs
    (default: 10).
* `--exp`: A string argument specifying the name of the experiment
    (default: "default_exp"
* `--save_path`: A string argument specifying the name and location to
    save trained model.
"""

import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

from train import train
from model import Vgg16, Resnet18
from evaluate_model import evaluate_model
from dataset import ImageDataLoader

# parse commandline arguments
parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=10,
                    help='Number of training epochs.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Number of training epochs.')
parser.add_argument('--exp', type=str, default='default_exp',
                    help='Experiment name.')
parser.add_argument('--save_path', type=str, default=None,
                    help='Path to save model')
parser.add_argument('--model_name', type=str, default='resnet',
                    help='Name of CNN model to train')

args = parser.parse_args()
EPOCHS = int(args.epochs)
lr = int(args.lr)
experiment_name = args.exp
save_path = args.save_path
MODEL_NAME = args.model_name

# choose device to train model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(28)

# define number of classes
NUM_CLASSES = 3


def get_model(model_name):
    '''
    Factory method to create model instance using model name
    '''
    if model_name == 'vgg':
        model = Vgg16(NUM_CLASSES)
    elif model_name == 'resnet':
        model = Resnet18(NUM_CLASSES)
    else:
        print('Invalid model name')
        return None

    return model


if __name__ == "__main__":
    # define dataset path
    TRAIN_DIR = 'dataset/final/train'
    VAL_DIR = 'dataset/final/val'

    # get model and data loaders

    model = get_model(MODEL_NAME)
    data_loader = ImageDataLoader(TRAIN_DIR, VAL_DIR)
    train_loader, val_loader, doc_classes = data_loader.get_loader()

    # tensorboard logger for experimentation tracking
    writer = SummaryWriter(log_dir=f'runs/{experiment_name}')

    # train model
    train_acc, train_loss, val_acc, val_loss = train(
        model, EPOCHS, train_loader, val_loader, writer,
        lr=lr, device=device, save_file=save_path)

    # evaluate model
    true_train, preds_train = evaluate_model(
        model, train_loader, doc_classes, writer, 'Train', device, )
    true_val, preds_val = evaluate_model(
        model, val_loader, doc_classes, writer, 'Val', device)
