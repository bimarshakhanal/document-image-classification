"""
Module with functions for classification model evaluation
"""

import torch
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

from base_logger import logger


def evaluate_model(model, data_loader, doc_classes, writer, ds, device='cpu'):
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
    report = classification_report(true,
                                   preds, target_names=doc_classes,
                                   output_dict=True)
    # writer.add_hparams(f'{ds}-result', report['macro avg'])
    # accuracy = accuracy_score(true, preds)
    # print(f"{ds} Accuracy: ", accuracy)
    logger.info("%s set evaluated with accuracy score of %f",
                ds, report['accuracy'])

    plot_confusion_matrix(preds, true, doc_classes, writer, ds)

    return {**report['macro avg'], "accuracy": report['accuracy']}


def plot_confusion_matrix(preds, targets, doc_classes, writer, ds):
    """
    Plots and logs a confusion matrix to TensorBoard.
    Args:
        preds: A list containing the predicted labels for the evaluated data.
        targets: A list containing the true labels for the evaluated data.
        doc_classes: A list containing the names of the document classes.
        writer: A TensorBoard SummaryWriter object for logging the confusion matrix.
        ds: An optional string to specify the dataset name (e.g., "train", "val", "test")
    """

    # calculate confusion matrix
    conf_mat = confusion_matrix(preds, targets)

    # create confusion matrix figure
    ax = plt.subplot()
    sn.heatmap(conf_mat, annot=True, fmt="g")
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(f'Confusion Matrix-{ds} Set', fontsize=10)
    ax.xaxis.set_ticklabels(doc_classes, rotation=90, fontsize=7)
    ax.yaxis.set_ticklabels(doc_classes, rotation=0, fontsize=7)

    # log confusion matrix figure to tensorboard
    writer.add_figure(tag=f'confusion_matrix-{ds}', figure=ax.get_figure())
    logging.info('%s Confusion matrix logged to tensorboard', ds)
