import os
from pathlib import Path
from typing import Optional

import conllu
import datasets
import more_itertools as mit
import requests
import stanza
from datasets import DatasetDict, Sequence, Value
from github import Github
from tango import Step
from tango.common import Tqdm
from tango.integrations.datasets import DatasetsFormat


@Step.register("microbert2.data.conllu::read_conllu")
class ReadConllu(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT = DatasetsFormat()

    def run(
        self,
        train_path: str,
        dev_path: str,
    ) -> DatasetDict:
        features = datasets.Features(
            {
                "tokens": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
                "xpos": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
                "head": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
                "deprel": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            }
        )

        def tokenlist_to_record(tl: conllu.TokenList):
            tokens = []
            xpos = []
            head = []
            deprel = []
            for t in tl:
                if isinstance(t["id"], int):
                    tokens.append(t["form"])
                    xpos.append(t["xpos"])
                    head.append(t["head"])
                    deprel.append(t["deprel"])
            return {
                "tokens": tokens,
                "xpos": xpos,
                "head": head,
                "deprel": deprel,
            }

        def generator(filepath):
            def inner():
                with open(filepath, "r") as f:
                    for x in conllu.parse_incr(f):
                        yield tokenlist_to_record(x)

            return inner

        train_dataset = datasets.Dataset.from_generator(generator(train_path), features=features)
        dev_dataset = datasets.Dataset.from_generator(generator(dev_path), features=features)
        self.logger.info(f"First train sentence: {train_dataset[0]}")
        self.logger.info(f"First dev sentence: {dev_dataset[0]}")

        return DatasetDict({"train": train_dataset, "dev": dev_dataset})


@Step.register("microbert2.data.conllu::read_text_only_conllu")
class ReadTextOnlyConllu(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT = DatasetsFormat()

    PATH_MAP = {
        "coptic": "data/coptic/converted",
        "english": "data/english/converted",
        "greek": "data/greek/converted",
        "indonesian": "data/indonesian/converted_punct",
        "maltese": "data/maltese/converted_punct",
        "tamil": "data/tamil/converted_punct",
        "uyghur": "data/uyghur/converted_punct",
        "wolof": "data/wolof/converted_punct",
    }

    def run(
        self,
        stanza_retokenize: bool = False,
        stanza_language_code: Optional[str] = None,
        shortcut: Optional[str] = None,
        conllu_path_train: Optional[str] = None,
        conllu_path_dev: Optional[str] = None,
        stanza_use_mwt: bool = True,
    ) -> DatasetDict:
        if stanza_retokenize:
            config = {
                "processors": "tokenize,mwt" if stanza_use_mwt else "tokenize",
                "lang": stanza_language_code,
                "use_gpu": True,
                "logging_level": "INFO",
                "tokenize_pretokenized": False,
                "tokenize_no_ssplit": True,
            }
            pipeline = stanza.Pipeline(**config)

        def retokenize(sentences, path):
            batch_size = 256

            space_separated = [" ".join(ts) for ts in sentences]
            chunks = list(mit.chunked(space_separated, batch_size))

            outputs = []
            for chunk in Tqdm.tqdm(chunks, desc=f"Retokenizing {path} with Stanza..."):
                output = pipeline("\n\n".join(chunk))
                for sentence in output.sentences:
                    s = sentence.to_dict()
                    retokenized = [t["text"] for t in s]
                    outputs.append(retokenized)
            for old, new in zip(sentences, outputs):
                if len(old) != len(new):
                    self.logger.debug(f"Retokenized sentence from {len(old)} to {len(new)}:\n\t{old}\n\t{new}\n")
            return outputs

        def read_conllu(path):
            with open(path, "r") as f:
                sentences = [[t["form"] for t in s] for s in conllu.parse_incr(f)]
                if stanza_retokenize:
                    sentences = retokenize(sentences, path)
                return sentences

        if shortcut is not None:
            if shortcut not in ReadTextOnlyConllu.PATH_MAP:
                raise ValueError(f"Unrecognized shortcut: {shortcut}")
            self.logger.info(f"Recognized shortcut {shortcut}")
            conllu_path_train = ReadTextOnlyConllu.PATH_MAP[shortcut] + os.sep + "train"
            conllu_path_dev = ReadTextOnlyConllu.PATH_MAP[shortcut] + os.sep + "dev"
            self.logger.info(f"Train path set to {conllu_path_train}")
            self.logger.info(f"Dev path set to {conllu_path_dev}")

        train_docs = (
            [read_conllu(conllu_path_train)]
            if conllu_path_train.endswith(".conllu")
            else [read_conllu(f) for f in Path(conllu_path_train).glob("**/*.conllu") if f.is_file()]
        )
        self.logger.info(
            f"Loaded {len(train_docs)} training docs from {conllu_path_train} "
            f"containing {len([t for d in train_docs for s in d for t in s])} tokens."
        )
        dev_docs = (
            [read_conllu(conllu_path_dev)]
            if conllu_path_dev.endswith(".conllu")
            else [read_conllu(f) for f in Path(conllu_path_dev).glob("**/*.conllu") if f.is_file()]
        )
        self.logger.info(
            f"Loaded {len(dev_docs)} dev docs from {conllu_path_dev} "
            f"containing {len([t for d in dev_docs for s in d for t in s])} tokens."
        )

        train_dataset = datasets.Dataset.from_list(
            [{"tokens": s} for d in train_docs for s in d],
            features=datasets.Features({"tokens": datasets.Sequence(datasets.Value(dtype="string"))}),
        )
        dev_dataset = datasets.Dataset.from_list(
            [{"tokens": s} for d in dev_docs for s in d],
            features=datasets.Features({"tokens": datasets.Sequence(datasets.Value(dtype="string"))}),
        )

        self.logger.info(f"First train sentence: {train_dataset[0]}")
        self.logger.info(f"First dev sentence: {dev_dataset[0]}")

        return DatasetDict({"train": train_dataset, "dev": dev_dataset})


@Step.register("microbert2.data.conllu::read_ud_treebank")
class ReadUDTreebank(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT = DatasetsFormat()

    REPO_MAP = {
        "coptic": "UD_Coptic-Scriptorium",
        "english": "UD_English-GUM",
        "greek": "UD_Ancient_Greek-PROIEL",
        "indonesian": "UD_Indonesian-GSD",
        "maltese": "UD_Maltese-MUDT",
        "tamil": "UD_Tamil-TTB",
        "uyghur": "UD_Uyghur-UDT",
        "wolof": "UD_Wolof-WTB",
    }

    def run(
        self,
        shortcut: Optional[str] = None,
        repo: Optional[str] = None,
        tag: str = "r2.11",
    ) -> DatasetDict:
        api = Github()
        if shortcut is not None:
            if shortcut not in ReadUDTreebank.REPO_MAP:
                raise ValueError(f"Unrecognized shortcut: {shortcut}")
            repo = ReadUDTreebank.REPO_MAP[shortcut]
        r = api.get_repo(f"UniversalDependencies/{repo}")
        all_tags = r.get_tags()
        filtered_tags = [t for t in all_tags if t.name == tag]
        if len(filtered_tags) == 0:
            raise ValueError(f"Requested tag {tag} was not found. Available tags: {all_tags}")

        files = [
            (f.path, f.download_url)
            for f in r.get_contents("/", ref=filtered_tags[0].commit.sha)
            if f.path.endswith(".conllu")
        ]
        if len(files) != 3:
            raise ValueError("Repositories without a train, dev, and test split are not supported.")
        train_url = [url for name, url in files if "train.conllu" in name][0]
        dev_url = [url for name, url in files if "dev.conllu" in name][0]
        test_url = [url for name, url in files if "test.conllu" in name][0]

        train_conllu = conllu.parse(requests.get(train_url).text)
        dev_conllu = conllu.parse(requests.get(dev_url).text)
        test_conllu = conllu.parse(requests.get(test_url).text)

        def tokenlist_to_record(tl: conllu.TokenList):
            # filter out supertokens and ellipsis tokens
            metadata = tl.metadata
            tl = [t for t in tl if isinstance(t["id"], int)]
            return {
                "tokens": [t["form"] for t in tl],
                "xpos": [t["xpos"] for t in tl],
                "head": [t["head"] for t in tl],
                "deprel": [t["deprel"] for t in tl],
            }

        features = datasets.Features(
            {
                "tokens": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
                "xpos": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
                "head": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
                "deprel": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            }
        )

        train_dataset = datasets.Dataset.from_list([tokenlist_to_record(tl) for tl in train_conllu], features=features)
        dev_dataset = datasets.Dataset.from_list([tokenlist_to_record(tl) for tl in dev_conllu], features=features)
        test_dataset = datasets.Dataset.from_list([tokenlist_to_record(tl) for tl in test_conllu], features=features)
        self.logger.info(f"First train sentence: {train_dataset[0]}")
        self.logger.info(f"First dev sentence: {dev_dataset[0]}")

        return DatasetDict({"train": train_dataset, "dev": dev_dataset, "test": test_dataset})
