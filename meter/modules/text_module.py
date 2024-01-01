import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings, BertModel, BertEncoder, BertLayer
from .bert_model import BertCrossLayer, BertAttention
from . import heads, objectives, meter_utils
from .clip_model import build_model, adapt_position_encoding
from transformers import AutoModel

class TextTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.text_transformer = AutoModel.from_pretrained(config['tokenizer'])

        self.cross_modal_text_pooler = heads.Pooler(config["hidden_size"])
        self.cross_modal_text_pooler.apply(objectives.init_weights)

        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
        ):
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

        hs = self.hparams.config["hidden_size"]

        if self.hparams.config["loss_names"]["mnre"] > 0:
            self.mnre_classifier = nn.Sequential(
                nn.Linear(hs, hs),
                nn.LayerNorm(hs),
                nn.GELU(),
                nn.Linear(hs, 23),
            )
            self.mnre_classifier.apply(objectives.init_weights)

        # 设定模型需要监控的指标
        meet_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

    def infer(
        self,
        batch,
    ):
        text_ids = batch[f"text_ids"]
        text_masks = batch[f"text_masks"]

        # text_embeds = self.text_transformer.embeddings(input_ids=text_ids)
        # device = text_embeds.device
        # input_shape = text_masks.size()
        # extend_text_masks = self.text_transformer.get_extended_attention_mask(text_masks, input_shape, device)
        # for layer in self.text_transformer.encoder.layer:
        #     text_embeds = layer(text_embeds, extend_text_masks)[0]

        # 简化写法
        text_embeds = self.text_transformer(text_ids, text_masks).last_hidden_state if not self.hparams.config[
            "emb_only"] else self.text_transformer.embeddings(input_ids=text_ids)

        cls_feats = self.cross_modal_text_pooler(text_embeds)

        ret = {
            "cls_feats": cls_feats,
            "text_ids": text_ids,
            "text_masks": text_masks,
        }


        return ret

    def forward(self, batch):
        ret = dict()

        ret.update(objectives.compute_mnre(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        # meet_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        meet_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        # meet_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        meet_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        # meet_utils.set_task(self)
        output = self(batch)
        ret = dict()

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        meet_utils.epoch_wrapup(self)

    def configure_optimizers(self):  # 设定模型优化器和学习率
        return meet_utils.set_schedule(self)
