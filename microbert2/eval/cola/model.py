import logging
import os
from copy import copy
from pathlib import Path
from typing import Any, Dict, Optional

import torch.nn.functional as F
from tango.integrations.torch import Model
from tango.integrations.transformers import Tokenizer
from torchmetrics import Accuracy, MatthewsCorrCoef
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

logger = logging.getLogger(__name__)


@Model.register("microbert2.eval.cola.model::cola_model")
class ColaModel(Model):
    def __init__(self, model_path: str, tokenizer: Tokenizer, classifier_dropout: float = 0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_path)
        self.config.classifier_dropout = classifier_dropout
        self.config.num_labels = 1

        pooler_models = {"distilbert-base-cased", "bert-base-cased"}
        self.has_pooler = model_path in pooler_models
        if self.has_pooler:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            self.encoder = AutoModel.from_pretrained(model_path)
            self.classifier = RobertaClassificationHead(self.config)
        self.tokenizer = tokenizer

        if self.has_pooler:
            self.accuracy = Accuracy(task="multiclass", num_classes=2, top_k=1)
        else:
            self.accuracy = Accuracy(task="binary")
        self.mcc = MatthewsCorrCoef(2)

    def forward(
        self,
        input_ids,
        attention_mask,
        label=None,
    ):
        if self.has_pooler:
            outputs: SequenceClassifierOutput = self.model(input_ids, attention_mask)
            logits = outputs.logits
        else:
            encoder_outputs: BaseModelOutputWithPoolingAndCrossAttentions = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
            last_encoder_state = encoder_outputs.last_hidden_state
            logits = self.classifier(last_encoder_state).squeeze(-1)

        probs = logits.sigmoid()

        if label is not None:
            if self.has_pooler:
                loss = F.cross_entropy(logits, label)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, label.float())
            accuracy = self.accuracy(probs, label)
            mcc = self.mcc(probs, label)
            outputs = {"loss": loss, "accuracy": accuracy, "mcc": mcc}
            self.mcc.reset()
            self.accuracy.reset()
            return outputs
        else:
            return {}
