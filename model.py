"""
rlsn 2024
"""
from transformers import PreTrainedModel
from transformers.utils import ModelOutput
from transformers.models.vit.modeling_vit import ViTPooler, ViTEncoder
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch
import torch.nn as nn
import numpy as np

class Vit3DEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_patches = int(np.prod(np.array(config.image_size)/np.array(config.patch_size)))
        patch_dim = np.prod(config.patch_size)*config.num_channels
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))
        self.projection = nn.Conv3d(config.num_channels, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, pixel_values):
        batch_size, num_channels, depth, height, width = pixel_values.shape
        embeddings = self.projection(pixel_values).flatten(2).transpose(1,2)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class VitDet3D(PreTrainedModel):
    def __init__(self, config, add_pooling_layer = True):
        super().__init__(config)
        self.embeddings = Vit3DEmbeddings(config)
        self.encoder = ViTEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config) if add_pooling_layer else None
        self.classification_head = nn.Linear(config.hidden_size, config.num_labels)
        self.bbox_head = nn.Linear(config.hidden_size, 6)
        self.config = config

    def forward(self, pixel_values, labels=None, bbox=None):
        embeddings = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(embeddings)
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        logits = self.classification_head(pooled_output)
        bbox_pred = self.bbox_head(pooled_output)
        
        if labels is not None and bbox is not None:
            loss_bbox_fn = MSELoss(reduction='none')
            if self.config.num_labels == 1:
                loss_cls_fn = BCEWithLogitsLoss()
                loss = loss_cls_fn(logits.view(-1), labels.float())
            else:
                loss_cls_fn = CrossEntropyLoss()
                loss = loss_cls_fn(logits, labels)
          
            mask = labels.unsqueeze(-1).bool()
            mse_loss = loss_bbox_fn(bbox_pred, bbox)*mask
            loss += mse_loss.mean()
        else:
            loss = None

        return ModelOutput(
            loss=loss,
            logits=logits,
            bbox=bbox_pred,
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )