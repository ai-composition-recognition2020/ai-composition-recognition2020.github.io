import glob
import pretty_midi
import torch

from torch.utils.data import DataLoader, Dataset
from utils import files_to_vector_array, load_midi, logger
from sklearn.preprocessing import StandardScaler


class MidiDataSet(Dataset):
    def __init__(self, root_dir: str, cfg: dict, scaler: bool = False, test: bool = False):
        self.test = test
        self.names, self.midi_datas, self.labels = files_to_vector_array(root_dir, cfg, scaler, test)

        if self.midi_datas is None:
            raise RuntimeError("data is empty")
        self.len = len(self.midi_datas)
        self.midi_datas = torch.from_numpy(self.midi_datas)  # .float()

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.test:
            return self.names[index], self.midi_datas[index]
        else:
            return self.midi_datas[index], self.labels[index]
