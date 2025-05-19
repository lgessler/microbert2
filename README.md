# Introduction

**microbert2** is an enhanced reimplementation of [MicroBERT](https://github.com/lgessler/microbert), a utility for training BERT-like encoder-only Transformer models for low-resource languages.
For more information, please see [our paper](https://aclanthology.org/2022.mrl-1.9/).

# Use

## Quick Start

1. Install requirements:

```
pip install -r requirements.txt
```

2. Run the sample training configuration for Coptic, editing it as desired:

```
tango run configs/microbert_coptic.jsonnet
```

3. You can use tensorboard to monitor progress. Type `tensorboard --logdir workspace` to start the dashboard, and navigate to http://localhost:6006 to view training and validation curves.

4. The finished `transformers.AutoModel`- and `transformers.AutoTokenizer`-compatible models will be saved to `workspace/models`.
Be warned: multiple runs of the same configuration will result in the original model being overwritten, so be sure to e.g. change the value of `name` inside the configuration between runs.

## Application to Other Datasets

Use `configs/microbert_coptic.jsonnet` as a starting point, being sure to change key hyperparameters, tasks, and dataset paths as appropriate.
If you have questions or problems, please feel free to open an issue.

# Citing
If you'd like to cite our work, please use the following citation:

```
@inproceedings{gessler-zeldes-2022-microbert,
    title = "{M}icro{BERT}: Effective Training of Low-resource Monolingual {BERT}s through Parameter Reduction and Multitask Learning",
    author = "Gessler, Luke  and
      Zeldes, Amir",
    booktitle = "Proceedings of the The 2nd Workshop on Multi-lingual Representation Learning (MRL)",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.mrl-1.9",
    pages = "86--99",
}
```

