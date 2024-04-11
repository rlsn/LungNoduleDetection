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

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, 
            out_channels, 
            kernel_size=[3,3,3], 
            stride=stride, 
            padding=1,
            bias=False
        )
        self.downsample = downsample
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            out_channels, 
            out_channels, 
            kernel_size=[3,3,3], 
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return  out

class CNNFeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        patch_size = config.patch_size
        image_size = config.image_size
        self.in_channels = 128
        
        self.conv1 = nn.Conv3d(config.num_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(128, 2)
        self.layer2 = self._make_layer(256, 2, stride=2)
        self.layer3 = self._make_layer(512, 2, stride=2)

        # self.avgpool = nn.AdaptiveAvgPool3d([3, 8, 8])

    def _make_layer(self, num_channels, num_layers, stride = 1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, num_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(num_channels),
            )

        layers = []
        layers.append(ResBlock(self.in_channels, num_channels, stride, downsample))
        self.in_channels = num_channels
        for _ in range(1, num_layers):
            layers.append(ResBlock(self.in_channels, num_channels, 1))

        return nn.Sequential(*layers)

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.avgpool(x)
        x = x.flatten(2).transpose(1,2)
        return x

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