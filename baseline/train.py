import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rich.progress import BarColumn, Progress, TextColumn
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset

from dataset import MidiDataSet
from model import *
from utils import yaml_load, logger

progress = Progress(
    TextColumn("[bold blue]{task.fields[phase]}", justify="right"),
    TextColumn("[bold blue]{task.fields[epoch]}", justify="right"),
    TextColumn("[bold blue]{task.fields[loss]}", justify="left"),
    BarColumn(bar_width=None),
    "[progress.percentage]{task.percentage:>3.1f}%",
    TextColumn("[bold blue]{task.fields[per]}", justify="right")
)


def cal_loss(model, loss_func, x, y, opt=None):
    """
    Calculate loss and updates the gradient

    :param model nn.Module: model
    :param loss_func function: loss function
    :param x torch.Tensor: input data
    :param y torch.Tensor: labels
    :param opt nn.optimizer: optimizer
    """

    pred = model.forward(x)
    y = y.contiguous().view(-1)
    loss = loss_func(pred, y) #

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item()


def fit(device, epochs, model, loss_func, opt, train_dl, eval_dl=None):
    """
    fit
    According to the number of iterations, loss function for training

    :param device device: device
    :param epochs int: epochs num
    :param model nn.Module: model
    :param loss_func function: loss function
    :param opt nn.optimizer: optimizer
    :param train_dl DataLoader: data loader
    :param eval_dl DataLoader: data loader
    """
    logger.info(f"""epochs: {epochs}, model: {model.__class__.__name__},
                loss_func: {loss_func.__class__.__name__}, opt: {opt.__class__.__name__}""")

    with progress:
        for epoch in range(epochs):
            model.train()
            loss = 0
            len_of_dl = len(train_dl)
            task_train = progress.add_task("train", phase="TRAIN",
                                        epoch=f"Epoch: {epoch+1:>3}/{epochs:<3}",
                                        loss=f"[yellow]Loss: {loss: <8.4f}",
                                        per=f"1/{len_of_dl}",
                                        total=len_of_dl)

            for i, d in enumerate(train_dl):
                x, y = d
                loss = cal_loss(model, loss_func, x.to(device), x.to(device), opt)
                progress.update(task_train, advance=1,
                                loss=f"[yellow]Loss: {loss: <8.4f}",
                                per=f"{i+1}/{len_of_dl}")

            if eval_dl is not None:
                model.eval()
                eval_loss = 0
                len_of_dl = len(eval_dl)
                task_eval = progress.add_task("eval", phase="[red]EVAL  ",
                                            epoch=f"Epoch: {epoch+1:>3}/{epochs:<3}",
                                            loss=f"[yellow]Loss: {eval_loss: <8.4f}",
                                            per=f"1/{len_of_dl}",
                                            total=len_of_dl)

                with torch.no_grad():
                    for i, d in enumerate(eval_dl):
                        x, y = d

                        eval_loss = cal_loss(model, loss_func, x.to(device), x.to(device))
                        progress.update(task_eval, advance=1,
                                        loss=f"[yellow]Loss: {eval_loss: <8.4f}",
                                        per=f"{i+1}/{len_of_dl}")


if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load config from config.yaml
    config    = yaml_load("./config.yaml")
    train_cfg = config.get("train", {})
    model_cfg = config.get("model", {})
    base_cfg  = config.get("base", {})

    # config about train
    epochs     = train_cfg.get("epochs", 1)
    batch_size = train_cfg.get("batch_size", 16)
    shuffle    = train_cfg.get("shuffle", True)
    lr         = train_cfg.get("lr", 0.1)
    vs         = train_cfg.get("vs", 0)

    # config about model
    init_input     = model_cfg["init_input"]
    model_name     = model_cfg["model"]
    loss_func_name = model_cfg["loss_func"]
    optimizer_name = model_cfg["optimizer"]

    # base config
    dev_data_path   = base_cfg["dev_data"]

    # dataset and dataloader config
    dataset          = MidiDataSet(dev_data_path, base_cfg)
    train_dataloader = None
    eval_dataloader  = None

    # split training data to train and validate
    if vs != 0:
        val_size   = int(vs * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    else:
        train_dataset, eval_dataset = dataset, None

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    if eval_dataset is not None:
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    # config model
    model     = eval(model_name)(*init_input).to(device)
    loss_func = eval(f"F.{loss_func_name}")
    opt       = eval(f"optim.{optimizer_name}")(model.parameters(), lr=lr)

    # train
    fit(device, epochs, model, loss_func, opt, train_dataloader, eval_dataloader)

    # save model
    check_point = {
        "model_name": model_name,
        "input_size": init_input,
        "state_dict": model.state_dict(),
        "loss_func": loss_func_name,
    }
    torch.save(check_point, f"./model.pth")
    logger.info(f"save model to ./model.pth")
