"""
rlsn 2024
"""
import numpy as np
import torch
from dataset import LUNA16_Dataset, collate_fn
from transformers import ViTConfig
from model import VitDet3D
from transformers import TrainingArguments, Trainer
from sklearn.metrics import f1_score

def iou_3d(bbox_pred,bbox):
    ilow = np.maximum(bbox_pred,bbox)[:,:3]
    ihigh = np.minimum(bbox_pred,bbox)[:,3:]
    i_sides = np.maximum(ihigh-ilow,0)
    i_vol = np.prod(i_sides,-1)
    o_vol = np.prod(bbox_pred[:,3:]-bbox_pred[:,:3],-1)+np.prod(bbox[:,3:]-bbox[:,:3],-1)-i_vol
    return (i_vol/o_vol).mean()

def compute_metrics(eval_pred):
    predictions, groundtruth = eval_pred
    logits = predictions[0]
    labels = groundtruth[0]

    mask = labels.astype(bool)
    bbox_pred = predictions[1][mask]
    bbox = groundtruth[1][mask]

    preds = (logits>0).astype(int)
    f1 = f1_score(labels, preds)
    if bbox.shape[0]>0:
        iou = iou_3d(bbox_pred,bbox)
    else:
        iou = 1.0
    return dict(f1=f1, iou=iou)

def train(data_dir, log_dir, model_dir=None, resume=True):
    config = ViTConfig.from_pretrained("model_config.json")
    print(config)
    
    valid_split = [9]
    train_split = np.arange(9)

    print("preparing datasets")
    train_dataset = LUNA16_Dataset(split = train_split, data_dir=data_dir, crop_size=config.image_size, patch_size=config.patch_size, samples_per_img=16)
    valid_dataset = LUNA16_Dataset(split = valid_split, data_dir=data_dir, crop_size=config.image_size, patch_size=config.patch_size, samples_per_img=16)

    print("preparing model")
    model = VitDet3D(config)

    if model_dir is None:
        model_dir = "luna-train"

    args = TrainingArguments(
        model_dir,
        save_strategy="steps",
        evaluation_strategy="steps",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        max_steps=1000000,
        weight_decay=0.01,
        eval_steps=5000,
        logging_steps=200,
        save_steps=5000,
        save_total_limit=5,
        dataloader_num_workers=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=True,
        label_names=["labels","bbox"],
        logging_dir=log_dir,
        remove_unused_columns=False,
    )
    print(args)
    
    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )
    print("commence training")
        
    trainer.train(resume_from_checkpoint=resume)

if __name__=="__main__":
    train(data_dir="datasets/luna16", log_dir="logs", model_dir="luna-train", resume=False)