import time
from functools import partial
from typing import Any, Dict, List

import conllu
import datasets
import pandas as pd
import torch
from tango import JsonFormat, Step
from tango.common import Tqdm
from tango.common.dataset_dict import DatasetDict
from tango.integrations.datasets import DatasetsFormat
from tango.integrations.torch import EvalCallback, Model
from tango.integrations.transformers import Config, DataCollator
from tango.integrations.transformers.tokenizer import Tokenizer
from torchmetrics import Accuracy
from transformers import AutoModelForSequenceClassification
from transformers.models.auto.auto_factory import _get_model_class
from transformers.models.distilbert.modeling_distilbert import DistilBertForSequenceClassification


def conllu_to_dfs(conllu_path):
    def read_conllu(path):
        with open(path, "r") as f:
            return conllu.parse(f.read())

    def sentence_to_stype_instance(sentence: conllu.TokenList) -> Dict[str, Any]:
        stype = sentence.metadata["s_type"]
        text = sentence.metadata["text"]
        return {"label": stype, "text": text}

    records = [sentence_to_stype_instance(s) for s in read_conllu(conllu_path)]
    return pd.DataFrame.from_records(records)


def dfs_to_datasets(df, labels, tokenizer):
    tokenizer_records = [tokenizer(x) for x in df["text"]]
    df["input_ids"] = [x["input_ids"] for x in tokenizer_records]
    df["attention_mask"] = [x["input_ids"] for x in tokenizer_records]
    del df["text"]
    dataset = datasets.Dataset.from_pandas(
        df,
        features=datasets.Features(
            {
                "label": datasets.ClassLabel(names=labels),
                "input_ids": datasets.Sequence(datasets.Value(dtype="int32")),
                "attention_mask": datasets.Sequence(datasets.Value(dtype="int32")),
            }
        ),
    )

    return dataset


@Model.register("demo_auto_model_wrapper::from_config", constructor="from_config")
class AutoModelForSequenceClassificationWrapper(AutoModelForSequenceClassification):
    @classmethod
    def from_config(cls, config: Config, **kwargs) -> Model:
        model_class = _get_model_class(config, cls._model_mapping)
        model = model_class._from_config(config, **kwargs)

        def forward(self, *args, **kwargs):
            output = model_class.forward(self, *args, **kwargs)
            labels = kwargs.pop("labels")
            acc = Accuracy().to(model.device)
            if labels is not None:
                preds = output.logits.max(1).indices.to(model.device)
                labels = labels.to(model.device)
                output = dict(output)
                output["accuracy"] = acc(preds, labels)
            return output

        model.forward = forward.__get__(model)
        return model


@Step.register("construct_stype_instances")
class ConstructStypeInstances(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT = DatasetsFormat()

    def run(
        self,
        train_conllu: str,
        dev_conllu: str,
        test_conllu: str,
        tokenizer: Tokenizer,
        num_workers: int = 1,
    ) -> DatasetDict:
        dfs = {k: conllu_to_dfs(p) for k, p in zip(["train", "dev", "test"], [train_conllu, dev_conllu, test_conllu])}
        labels = dfs["train"].label.unique().tolist()
        dataset = datasets.DatasetDict({k: dfs_to_datasets(df, labels, tokenizer) for k, df in dfs.items()})

        self.logger.info(dataset["train"][0])

        return dataset.with_format("torch")


@Step.register("label_count")
class LabelCount(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT = JsonFormat()

    def run(self, dataset: DatasetDict):
        return len(dataset["train"].unique("label"))
