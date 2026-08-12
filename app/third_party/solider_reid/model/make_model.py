"""Minimal SOLIDER-REID transformer builder (Swin only, inference-ready)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .backbones.swin_transformer import (
    swin_base_patch4_window7_224,
    swin_small_patch4_window7_224,
    swin_tiny_patch4_window7_224,
)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


__factory_T_type = {
    "swin_base_patch4_window7_224": swin_base_patch4_window7_224,
    "swin_small_patch4_window7_224": swin_small_patch4_window7_224,
    "swin_tiny_patch4_window7_224": swin_tiny_patch4_window7_224,
}


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, semantic_weight):
        super().__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.reduce_feat_dim = cfg.MODEL.REDUCE_FEAT_DIM
        self.feat_dim = cfg.MODEL.FEAT_DIM
        self.dropout_rate = cfg.MODEL.DROPOUT_RATE

        convert_weights = True if pretrain_choice == "imagenet" else False
        img_size = cfg.INPUT.SIZE_TRAIN
        if isinstance(img_size, list):
            img_size = tuple(img_size)
        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](
            img_size=img_size,
            drop_path_rate=cfg.MODEL.DROP_PATH,
            drop_rate=cfg.MODEL.DROP_OUT,
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
            pretrained=model_path if model_path else None,
            convert_weights=convert_weights,
            semantic_weight=semantic_weight,
        )
        if model_path:
            self.base.init_weights(model_path)
        self.in_planes = self.base.num_features[-1]
        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        if self.reduce_feat_dim:
            self.fcneck = nn.Linear(self.in_planes, self.feat_dim, bias=False)
            self.fcneck.apply(weights_init_xavier)
            self.in_planes = cfg.MODEL.FEAT_DIM
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x, label=None, cam_label=None, view_label=None):
        global_feat, featmaps = self.base(x)
        if self.reduce_feat_dim:
            global_feat = self.fcneck(global_feat)
        feat = self.bottleneck(global_feat)
        if self.training:
            feat_cls = self.dropout(feat)
            cls_score = self.classifier(feat_cls)
            return cls_score, global_feat, featmaps
        if self.neck_feat == "after":
            return feat, featmaps
        return global_feat, featmaps

    def load_param(self, trained_path):
        raw = torch.load(trained_path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict):
            if "state_dict" in raw:
                param_dict = raw["state_dict"]
            elif "model" in raw:
                param_dict = raw["model"]
            else:
                param_dict = raw
        else:
            param_dict = raw
        loaded = 0
        for key, value in param_dict.items():
            name = key.replace("module.", "")
            if name not in self.state_dict():
                continue
            try:
                self.state_dict()[name].copy_(value)
                loaded += 1
            except Exception:
                continue
        if loaded <= 0:
            raise RuntimeError(f"No SOLIDER weights loaded from {trained_path}")
        return loaded


def make_model(cfg, num_class, camera_num, view_num, semantic_weight):
    if cfg.MODEL.NAME != "transformer":
        raise ValueError(f"Unsupported SOLIDER MODEL.NAME={cfg.MODEL.NAME!r}")
    if cfg.MODEL.JPM:
        raise ValueError("SOLIDER JPM local transformer is not vendored for Pass2")
    return build_transformer(
        num_class, camera_num, view_num, cfg, __factory_T_type, semantic_weight
    )
