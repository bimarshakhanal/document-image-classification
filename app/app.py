import torch

from train import train
from model import ClassificationModel
from evaluate_model import evaluate_model
from dataset import ImageDataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(28)


if __name__ == "__main__":
    TRAIN_DIR = 'dataset/final/train'
    VAL_DIR = 'dataset/final/val'

    model = ClassificationModel(3)
    data_loader = ImageDataLoader(TRAIN_DIR, VAL_DIR)

    train_loader, val_loader, doc_classes = data_loader.get_loader()
    train_acc, train_loss, val_acc, val_loss = train(
        model, 3, train_loader, val_loader, device=device)
    evaluate_model(model, train_loader, doc_classes, 'Train', device)
    evaluate_model(model, val_loader, doc_classes, 'Val', device)
