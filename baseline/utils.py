import csv
import glob
import math
import pickle
import time
import logging
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pretty_midi
import torch
import yaml
from sklearn.preprocessing import StandardScaler


logging.basicConfig(level=logging.DEBUG, filename="demo.log")
logger = logging.getLogger(' ')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def load_midi(midi_file: str, config: dict):
    """
    read midi file and translate it to vector

    :param midi_file str: [midi file path]
    :param config dict: [config dict]
    """
    midi = pretty_midi.PrettyMIDI(midi_file)
    piano_roll = midi.get_piano_roll(fs=8) # shape: 128, n_columns
    data = np.argmax(piano_roll[:, :config.get("seq_len", 128)], axis=0) # shape: 128,

    return data.reshape(1, -1) # shape: 1, 128


def files_to_vector_array(folder: str, config: dict, scaler:bool=False, test: bool=False):
    """
    read midi files from folder and translate them all to vector

    :param fold str: [midi file folder]
    :param config dict: [config dict]
    :param scaler bool: [use scaler]
    :param test bool: [whether the test]
    """

    labels = []
    names = []
    datas = None

    if test:
        for midi_file in glob.glob(f"{folder}/*"):
            data = load_midi(midi_file, config)
            if datas is None:
                datas = data
            else:
                datas = np.concatenate((datas, data), 0)

            names.append(midi_file.split("/")[-1])
    else:
        for midi_file in glob.glob(f"{folder}/fake/*"):
            data = load_midi(midi_file, config) # 1, 128
            if datas is None:
                datas = data
            else:
                datas = np.concatenate((datas, data), 0)

            labels.append(0)

        for midi_file in glob.glob(f"{folder}/real/*"):
            data = load_midi(midi_file, config)
            if datas is None:
                datas = data
            else:
                datas = np.concatenate((datas, data), 0)

            labels.append(1)

    if scaler:
        sc = StandardScaler().fit(datas)
        datas = sc.transform(datas)

    # names len: files number
    # datas shape: files number, 128
    # labels len: files number
    return names, datas, labels


def format_time():
    return time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())


def yaml_load(config_path: str) -> dict:
    with open(config_path) as f:
        param = yaml.safe_load(f)
    return param


def create_fold(fd):
    if type(fd) == str:
        fd = Path(fd)
    try:
        fd.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise e


def save_csv(save_file_path: str, save_data: list):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)
