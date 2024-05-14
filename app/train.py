"""
Module to train pytorch model for image classification
"""

import torch
from torch import nn
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler

import numpy as np
from tqdm import tqdm


def train(model, epochs, train_loader, val_loader, writer, lr=0.001,
          device='cpu', save_file=None):
    """
    Trains a PyTorch classification model on a given dataset.
    Args:
        model: The PyTorch model to be trained.
        epochs: The number of training epochs.
        train_loader: A PyTorch data loader that yields batches of training data.
        val_loader: A PyTorch data loader that yields batches of validation data.
        lr: The learning rate for the optimizer (defaults to 0.001).
        device: The device to use for training ('cpu' or 'cuda').
        save_file: An optional string specifying the filename to save the
                    trained model, won't save model if file name not given.

    Returns:
      Four lists containing the training and validation losses and
      accuracies for each epoch:
          - train_acc: A list of average training accuracies for each epoch.
          - train_loss: A list of average training losses for each epoch.
          - val_acc: A list of average validation accuracies for each epoch.
          - val_loss: A list of average validation losses for each epoch.
    """

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.LinearLR(
        optimizer, start_factor=.5, end_factor=0.05, total_iters=epochs)

    model = model.to(device)

    train_acc = []
    train_loss = []
    val_loss = [99]
    val_acc = [0]

    for epoch in range(epochs):
        model.train(True)
        pbar = tqdm(train_loader)

        batch_loss = []
        batch_acc = []

        for i, data in enumerate(pbar):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs)
            acc = (labels == outputs.argmax(dim=-1)).float().mean().item()
            # Compute the loss and its gradients
            loss = criterion(outputs, labels)
            loss.backward()

            batch_loss.append(loss.item())
            batch_acc.append(acc)
            # Adjust learning weights
            optimizer.step()
            scheduler.step()

            pbar.set_description(f"Epoch: {epoch + 1}/{epochs} "
                                 f"Train Loss:{
                                     round(np.mean(batch_loss), 3)}; "
                                 f"Train Acc: {round(np.mean(batch_acc), 3)}; "
                                 f"Val Loss: {round(np.mean(val_loss), 3)}; "
                                 f"Val Acc: {round(np.mean(val_acc), 3)} ")

        model.eval()
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            val_batch_loss = []
            val_batch_acc = []
            for i, vdata in enumerate(val_loader):
                vinputs, vlabels = vdata
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)
                voutputs = model(vinputs)
                test_acc = (vlabels == voutputs.argmax(
                    dim=-1)).float().mean().item()
                vloss = criterion(voutputs, vlabels).item()

                val_batch_loss.append(vloss)
                val_batch_acc.append(test_acc)

        val_epoch_loss = np.mean(val_batch_loss)
        val_epoch_acc = np.mean(val_batch_acc)

        train_epoch_loss = np.mean(batch_loss)
        train_epoch_acc = np.mean(batch_acc)

        val_loss.append(val_epoch_loss)
        val_acc.append(np.mean(val_epoch_acc))
        train_acc.append(train_epoch_loss)
        train_loss.append(train_epoch_loss)

        writer.add_scalars(main_tag="Loss",
                           tag_scalar_dict={"train_loss": train_epoch_loss,
                                            "val_loss": val_epoch_loss},
                           global_step=epoch)
        writer.add_scalars(main_tag="Accuracy",
                           tag_scalar_dict={"train_acc": train_epoch_acc,
                                            "val_acc": val_epoch_acc},
                           global_step=epoch)

    if save_file:
        torch.save(model, save_file)
    writer.close()
    return train_acc, train_loss, val_acc, val_loss
