"""
Module with functions for classification model evaluation
"""
import logging

import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, filename='log/log.txt')


def evaluate_model(model, data_loader, doc_classes, ds="", device='cpu'):
    """
    Evaluates a classification model on a given dataset.
    Args:
        model: The PyTorch model to be evaluated.
        data_loader: A PyTorch data loader that yields batches of data.
        doc_classes: A list containing the names of the document classes.
        ds: An optional string to specify the dataset name (train/val/test)
        device: The device to use for evaluation ('cpu' or 'cuda').

    Returns:
        A tuple containing two lists:
            - true_labels: A list containing the true labels.
            - predicted_labels: A list containing the predicted class indices.
    """
    print(f"[INFO] evaluating network on {ds} set...")
    # turn off autograd for testing evaluation
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()
        # initialize a list to store our predictions
        preds = []
        true = []
        # loop over the dataset
        pbar = tqdm(data_loader)
        for (x, y) in pbar:
            # send the input to the device
            x = x.to(device)
            # make the predictions and add them to the list
            true.extend(y.cpu().tolist())
            pred = model(x)
            preds.extend(pred.argmax(axis=1).cpu().tolist())
    # generate a classification report
    print(classification_report(true,
                                preds, target_names=doc_classes))
    accuracy = accuracy_score(true, preds)
    print(f"{ds} Accuracy: ", accuracy)
    logger.info("%s set evaluated with accuracy score of %f", ds, accuracy)
    return true, preds
