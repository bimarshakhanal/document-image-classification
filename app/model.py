"""
PyTorch CNN model for image classification
"""
from torchvision.models import resnet18
from torch import nn


class ClassificationModel(nn.Module):
    """
    A basic image classification model based on the pre-trained ResNet-18
    architecture.
    This class takes an image as input, passes it through a pre-trained
    ResNet-18 model with its final fully-connected layer replaced to
    accommodate the number of output classes.

    Args:
        num_classes: The number of classes to classify images into.
    """

    def __init__(self, num_classes):
        """
        Initializes the classification model.

        Args:
            num_classes: The number of classes to classify images into.
        """
        super(ClassificationModel, self).__init__()
        self.model = resnet18(weights="DEFAULT")
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x: Input tensor representing an image.

        Returns:
            A tensor representing the logits (unnormalized scores)
            for each class.
        """
        return self.model(x)
