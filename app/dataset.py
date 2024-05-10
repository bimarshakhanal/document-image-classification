'''
Prepare and load data loaders for batch training
'''

from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torch.utils.data import DataLoader


class ImageDataLoader:
    '''
    Class to prepare and load document image dataset
    '''

    def __init__(self, train_dir, val_dir, batch_size):
        # image dimension
        self.img_height = 224
        self.img_width = 224
        self.batch_size = batch_size
        self.train_dir = train_dir
        self.val_dir = val_dir

        self.transform_train = v2.Compose([
            v2.ToPILImage(),
            v2.Resize((self.img_width, self.img_height)),
            v2.ColorJitter(brightness=.5, hue=.3),
            v2.RandomGrayscale(p=0.3),
            v2.ToTensor(),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transform_val = v2.Compose([
            v2.Resize((self.img_width, self.img_height)),
            v2.ToPILImage(),
            v2.ToTensor(),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_loader(self):
        '''
        Get train and validation loader
        Returns
            batched torch dataloader for train and validation data
        '''

        train_dataset = ImageFolder(
            root=self.train_dir, transform=self.transform_train)
        val_dataset = ImageFolder(
            root=self.val_dir, transform=self.transform_val)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True)

        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        return train_loader, val_loader, train_dataset.classes
