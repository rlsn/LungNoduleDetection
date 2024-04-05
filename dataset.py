"""
rlsn 2024
"""
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

id2label = {0: "NT", 1: "CLE", 2: "PSE", 3: "PLE"}
label2id = {"NT":0, "CLE":1, "PSE":2, "PLE":3}
data_len = 115

def read_tiff(fn):
    im = cv2.imread(fn, -1)
    x = np.array(im)
    return x.astype(float)

def read_csv(fn):
    with open(fn,"r") as f:
        lines = [l.strip().split(",") for l in f.readlines()]
    return lines

class CTED_Dataset(Dataset):
    """
    https://lauge-soerensen.github.io/emphysema-database/
    """
    def __init__(self, split=None, transform=None):
        label_csv = read_csv("cted/slice_labels.csv")
        severity_csv = read_csv("cted/slice_severity.csv")

        slice_data = []
        label_data = []
        severity_data = []
        for (f1,label),(f2,severity) in zip(label_csv,severity_csv):
            im=read_tiff(f'cted/slices/{f1}.tiff')
            slice_data.append(im)
            label_data.append(int(label)-1)
            severity_data.append(int(severity))
        slice_data = np.array(slice_data)
        mean, std = slice_data.mean(), slice_data.std()
        self.slice_data = torch.tensor((slice_data-mean)/std).unsqueeze(1).to(torch.float)
        self.label_data = torch.tensor(label_data)
        self.severity_data = torch.tensor(severity_data)
        if split is not None:
            self.slice_data = self.slice_data[split]
            self.label_data = self.label_data[split]
            self.severity_data = self.severity_data[split]
        self.transform = transform
        
    def __len__(self):
        return len(self.slice_data)
        
    def __getitem__(self, idx):
        
        img = self.slice_data[idx]
        if self.transform is not None:
            img = self.transform(img)
        return {"pixel_value": img,
                "label": self.label_data[idx]}

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_value"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    # severity = torch.tensor([example["severity"] for example in examples])
    
    return {"pixel_values": pixel_values, "labels": labels}

