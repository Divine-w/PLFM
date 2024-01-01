import os
import copy
import torch
import numpy as np
import random
from tqdm import tqdm
from meter.config import ex
from meter.modules import METER
from meter.datasets import MNREDataset, MNREMarkedDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    AutoTokenizer
)
import warnings

warnings.filterwarnings('ignore')

def set_seed(seed=0):
    """set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def get_pretrained_tokenizer(from_pretrained):
    return AutoTokenizer.from_pretrained(from_pretrained)

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    set_seed(_config["seed"])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    test_dataset = MNREDataset(_config["data_root"],
                               (["default_val"] if len(_config["val_transform_keys"]) == 0 else _config[
                                   "val_transform_keys"]),
                               split="test",
                               image_size=_config["image_size"],
                               max_text_len=_config["max_text_len"],
                               draw_false_image=_config["draw_false_image"],
                               draw_false_text=_config["draw_false_text"],
                               image_only=_config["image_only"],
                               rel2id=_config["rel2id"],
                               device=device
                               )
    test_dataset.tokenizer = get_pretrained_tokenizer(_config["tokenizer"])
    test_data = DataLoader(
            test_dataset,
            batch_size=_config["per_gpu_batchsize"],
            shuffle=False,
            num_workers=_config["num_workers"],
            collate_fn=test_dataset.collate,
        )

    val_dataset = MNREDataset(_config["data_root"],
                               (["default_val"] if len(_config["val_transform_keys"]) == 0 else _config[
                                   "val_transform_keys"]),
                               split="val",
                               image_size=_config["image_size"],
                               max_text_len=_config["max_text_len"],
                               draw_false_image=_config["draw_false_image"],
                               draw_false_text=_config["draw_false_text"],
                               image_only=_config["image_only"],
                               rel2id=_config["rel2id"],
                               device=device
                               )
    val_dataset.tokenizer = get_pretrained_tokenizer(_config["tokenizer"])
    val_data = DataLoader(
        val_dataset,
        batch_size=_config["per_gpu_batchsize"],
        shuffle=False,
        num_workers=_config["num_workers"],
        collate_fn=val_dataset.collate,
    )

    train_dataset = MNREDataset(_config["data_root"],
                              (["default_val"] if len(_config["val_transform_keys"]) == 0 else _config[
                                  "val_transform_keys"]),
                              split="train",
                              image_size=_config["image_size"],
                              max_text_len=_config["max_text_len"],
                              draw_false_image=_config["draw_false_image"],
                              draw_false_text=_config["draw_false_text"],
                              image_only=_config["image_only"],
                              rel2id=_config["rel2id"],
                              device=device
                              )
    train_dataset.tokenizer = get_pretrained_tokenizer(_config["tokenizer"])
    train_data = DataLoader(
        train_dataset,
        batch_size=_config["per_gpu_batchsize"],
        shuffle=False,
        num_workers=_config["num_workers"],
        collate_fn=train_dataset.collate,
    )

    marked_dataset = MNREMarkedDataset(_config["data_root"],
                               (["default_val"] if len(_config["val_transform_keys"]) == 0 else _config[
                                   "val_transform_keys"]),
                               image_size=_config["image_size"],
                               max_text_len=_config["max_text_len"],
                               draw_false_image=_config["draw_false_image"],
                               draw_false_text=_config["draw_false_text"],
                               image_only=_config["image_only"],
                               rel2id=_config["rel2id"],
                               device=device
                               )
    marked_dataset.tokenizer = get_pretrained_tokenizer(_config["tokenizer"])
    marked_data = DataLoader(
        marked_dataset,
        batch_size=_config["per_gpu_batchsize"],
        shuffle=False,
        num_workers=_config["num_workers"],
        collate_fn=marked_dataset.collate,
    )

    model = METER(_config)
    model.to(device)
    model.eval()

    testfeats = torch.tensor([])
    with torch.no_grad():
        with tqdm(total=len(test_data), leave=False, dynamic_ncols=True) as pbar:
            pbar.set_description_str(desc="extract_testfeats")
            for batch in test_data:
                outputs = model(batch)
                testfeat = outputs["cls_feats"].detach().cpu()
                testfeats = torch.cat((testfeats, testfeat), 0)
    torch.save(testfeats, "test_feats.pt")

    valfeats = torch.tensor([])
    with torch.no_grad():
        with tqdm(total=len(val_data), leave=False, dynamic_ncols=True) as pbar:
            pbar.set_description_str(desc="extract_valfeats")
            for batch in val_data:
                outputs = model(batch)
                valfeat = outputs["cls_feats"].detach().cpu()
                valfeats = torch.cat((valfeats, valfeat), 0)
    torch.save(valfeats, "val_feats.pt")

    trainfeats = torch.tensor([])
    with torch.no_grad():
        with tqdm(total=len(train_data), leave=False, dynamic_ncols=True) as pbar:
            pbar.set_description_str(desc="extract_trainfeats")
            for batch in train_data:
                outputs = model(batch)
                trainfeat = outputs["cls_feats"].detach().cpu()
                trainfeats = torch.cat((trainfeats, trainfeat), 0)
    torch.save(trainfeats, "train_feats.pt")

    markedfeats = torch.tensor([])
    with torch.no_grad():
        with tqdm(total=len(marked_data), leave=False, dynamic_ncols=True) as pbar:
            pbar.set_description_str(desc="extract_markedfeats")
            for batch in marked_data:
                outputs = model(batch)
                markedfeat = outputs["cls_feats"].detach().cpu()
                markedfeats = torch.cat((markedfeats, markedfeat), 0)
    torch.save(markedfeats, "marked_feats.pt")