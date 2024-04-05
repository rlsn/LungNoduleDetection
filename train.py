"""
rlsn 2024
"""
import numpy as np
import torch
from dataset import id2label, label2id, data_len, CTED_Dataset, collate_fn
from transformers import ViTForImageClassification, ViTConfig
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize)


from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score

valid_size = 30

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return dict(accuracy=accuracy_score(predictions, labels))

if __name__=="__main__":
    config = ViTConfig.from_pretrained("model_config.json")
    config.id2label=id2label
    config.label2id=label2id
    print(config)

    _train_transforms = Compose([
                RandomResizedCrop(config.image_size),
                RandomHorizontalFlip()
            ])

    _val_transforms = Compose([
                Resize(config.image_size),
                CenterCrop(config.image_size)
            ])
    
    valid_split = np.random.choice(data_len, valid_size, replace=False)
    train_split = np.array([x for x in np.arange(data_len) if x not in valid_split])

    train_dataset = CTED_Dataset(split = train_split, transform=_train_transforms)
    valid_dataset = CTED_Dataset(split = valid_split, transform=_val_transforms)

    model = ViTForImageClassification(config)

    args = TrainingArguments(
        f"copd-train",
        save_strategy="steps",
        evaluation_strategy="steps",
        learning_rate=2e-5,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=10,
        num_train_epochs=300,
        weight_decay=0.01,
        logging_steps=500,
        save_steps=500,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir='logs',
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    trainer.train()