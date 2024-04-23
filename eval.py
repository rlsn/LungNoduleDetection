"""
rlsn 2024
"""
import numpy as np
import torch
from model import VitDet3D
from dataset import LUNA16_Dataset
from tqdm import tqdm

def l2norm(x):
    return np.sum(x**2,axis=-1,keepdims=True)**0.5    

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def to_coord(bbox, origin, space):
    d = len(bbox.shape)
    if d==1:
        bbox = np.expand_dims(bbox,0)
    space = space[::-1]
    origin = origin[::-1]
    center = (bbox[:,3:] + bbox[:,:3])/2*space + origin    
    center = center[:,::-1]
    diam = l2norm((bbox[:,3:] - bbox[:,:3])*space)/3**0.5
    coord = np.concatenate([center, diam],1)
    if d==1:
        coord = coord.flatten()
    return coord

def merge_cands(cands, merge_dist=10):
    # recursively merge candidates that locates within merge_dist from the group mean
    def merge(merge_list):
        new_list = [merge_list.pop(0)]
        end = True
        while len(merge_list):
            cand = merge_list.pop(0)
            mu = np.mean(cand,0)
            is_merged=False
            for i, ref in enumerate(new_list):
                ref_mu = np.mean(ref,0)
                dist = l2norm(mu[:3]-ref_mu[:3])
                if dist < merge_dist:
                    new_list[i]+=cand
                    is_merged=True
                    end = False
                    break
            if not is_merged:
                new_list.append(cand)
        return new_list, end
    merged_cands = [[c] for c in cands]
    end = False
    while not end:
        merged_cands, end = merge(merged_cands)
    return np.array([np.mean(c,0) for c in merged_cands])

def detect(model, sample, batch_size=32):
    candidates = []
    for i, pixel_values in enumerate(torch.split(sample["pixel_values"].to(model.device), batch_size)):
        img_shape = np.tile(np.array(pixel_values.shape[-3:]),2)
        offsets = np.tile(sample["offsets"][i*batch_size:(i+1)*batch_size],2)
        outputs = model(pixel_values=pixel_values)
     
        bbox = outputs.bbox.cpu().numpy()*img_shape+offsets
        coord = to_coord(bbox, sample["origin"], sample["space"])
        logits = outputs.logits.cpu().numpy()
        candidates.append(np.concatenate([coord,logits],1))

    candidates = np.concatenate(candidates,0)
    candidates = merge_cands(candidates)
    # threshold cutoff
    candidates = candidates[candidates[:,-1]>-5]
    return candidates

if __name__=="__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_path = "checkpoint/checkpoint-100000"
    output_path = "results.csv"
    model = VitDet3D.from_pretrained(model_path).eval().to(device)

    dataset = LUNA16_Dataset(data_dir="datasets/luna16").eval()

    with open(output_path,"w+",buffering=1) as wf:
        head = ["seriesuid","coordX","coordY","coordZ","probability"]
        wf.write(",".join(head)+"\n")
        with torch.no_grad():
            for sample in tqdm(dataset, total=len(dataset)):
                pred_coords = detect(model, sample)
                pred_coords = np.concatenate([pred_coords[:,:3],pred_coords[:,-1:]],-1).astype(str)
                uid = np.array([sample["uid"]]*len(pred_coords)).astype(str)
                uid = np.expand_dims(uid,1)
                re = np.concatenate([uid,pred_coords],1)
                for row in re:
                    wf.write(",".join(row)+"\n")