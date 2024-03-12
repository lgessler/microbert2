import itertools
from typing import Optional

import conllu
import datasets
from datasets import ClassLabel, Dataset, DatasetDict, Sequence, Value
from tango import Step
from tango.common import IterableDatasetDict, Tqdm
from tango.integrations.datasets import DatasetsFormat


@Step.register("microbert2.nahuatl.data::read_nahuatl_conllu")
class ReadNahuatlConllu(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT = DatasetsFormat()

    FILES = {
        "train": [
            "data/nahuatl/Book_01_-_The_Gods.conllu",
            "data/nahuatl/Book_02_-_Ceremonies.conllu",
            "data/nahuatl/Book_03_-_The_Origins_of_the_Gods.conllu",
            "data/nahuatl/Book_04_-_The_Art_of_Divination.conllu",
            "data/nahuatl/Book_06_-_Rhetoric_and_Moral_Philosophy.conllu",
            "data/nahuatl/Book_07_-_The_Sun,_Moon,_Stars,_and_the_Binding_of_the_Years.conllu",
            "data/nahuatl/Book_09_-_The_Merchants.conllu",
            "data/nahuatl/Book_10_-_The_People.conllu",
            "data/nahuatl/Book_11_-_Earthly_Things.conllu",
            "data/nahuatl/Book_12_-_The_Conquest_of_Mexico.conllu",
        ],
        "dev": [
            "data/nahuatl/Book_08_-_Kings_and_Lords.conllu",
        ],
        "test": [
            "data/nahuatl/Book_05_-_The_Omens.conllu",
        ],
    }

    def run(
        self,
    ) -> DatasetDict:

        def read_conllus(paths):
            sentences = []
            for path in paths:
                with open(path, "r") as f:
                    for s in conllu.parse_incr(f):
                        sentences.append({
                            "tokens": [t["form"] for t in s],
                            "xpos": [t["upos"] for t in s]
                        })
            return sentences

        train_sentences = read_conllus(ReadNahuatlConllu.FILES["train"])
        self.logger.info(
            f"Loaded {len(train_sentences)} training docs "
            f"containing {len([t for d in train_sentences for s in d for t in s])} tokens."
        )
        dev_sentences = read_conllus(ReadNahuatlConllu.FILES["dev"])
        self.logger.info(
            f"Loaded {len(dev_sentences)} dev docs "
            f"containing {len([t for d in dev_sentences for s in d for t in s])} tokens."
        )

        features = datasets.Features({
            "tokens": datasets.Sequence(datasets.Value(dtype="string")),
            "xpos": datasets.Sequence(datasets.Value(dtype="string", id=None), length=-1, id=None),
        })
        train_dataset = datasets.Dataset.from_list(train_sentences, features=features)
        dev_dataset = datasets.Dataset.from_list(dev_sentences, features=features)

        self.logger.info(f"First train sentence: {train_dataset[0]}")
        self.logger.info(f"First dev sentence: {dev_dataset[0]}")

        return DatasetDict({"train": train_dataset, "dev": dev_dataset})


@Step.register("microbert2.nahuatl.data::finalize")
class Finalize(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT = DatasetsFormat()
    
    def _get_labels(self, dataset):
        print(dataset[0])
        xpos = sorted(list(set(tag for s in dataset for tag in s["xpos"])))
        self.logger.info(f"Using xpos set: {xpos}")
        return xpos

    def run(
        self,
        dataset: IterableDatasetDict,
        treebank_dataset: IterableDatasetDict = None,
        unlabeled_per_labeled: int = 8,
    ) -> DatasetDict:
        dataset = dataset.remove_columns(["tokens"])
        print(dataset)

        new_dataset = {}
        for split, rows in dataset.items():

            def get_rows(rows, tree_is_gold=(0,)):
                def inner():
                    def process_row(v):
                        # - 2 because of CLS and SEP tokens
                        is_gold = "xpos" in v and not any(x in ["X", "", "_"] for x in v["xpos"])
                        num_whole_tokens = len(v["token_spans"]) // 2 - 2
                        return {
                            "input_ids": v["input_ids"],
                            "token_type_ids": v["token_type_ids"],
                            "attention_mask": v["attention_mask"],
                            "token_spans": v["token_spans"],
                            "head": [0] * num_whole_tokens,
                            "deprel": [0] * num_whole_tokens,
                            "xpos": v["xpos"] if is_gold else [0] * num_whole_tokens,
                            "tree_is_gold": [int(is_gold)],
                        }

                    for v in Tqdm.tqdm(rows, desc=f"Constructing {split}..", total=len(rows)):
                        yield process_row(v)

                return inner

            rows = list(get_rows(rows)())
            base_rows = [r for r in rows if not r["tree_is_gold"][0]]
            treebank_rows = [r for r in rows if r["tree_is_gold"][0]]
            xpos = ["NONE"]
            if len(treebank_rows) > 0:
                xpos += self._get_labels(treebank_rows)
                base_rows = list(base_rows)
                self.logger.info(f"Extending split {split} with gold treebanked sentences...")
                if split == "train":
                    target_length = len(list(base_rows)) // unlabeled_per_labeled
                    if len(treebank_rows) > target_length:
                        raise Exception("More treebank rows than expected!")
                    self.logger.info(f"{len(treebank_rows)} treebank sequences found. Repeating...")
                    treebank_rows = list(itertools.islice(itertools.cycle(treebank_rows), target_length))
                    self.logger.info(
                        f"Upsampled treebank instances to {len(treebank_rows)}. (Unlabeled: {len(base_rows)})"
                    )
                treebank_rows = treebank_rows

            features = datasets.Features(
                {
                    "input_ids": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
                    "attention_mask": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
                    "token_type_ids": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
                    "token_spans": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
                    "head": Sequence(feature=Value(dtype="int16", id=None), length=-1, id=None),
                    "deprel": Sequence(feature=ClassLabel(names=["NONE"], id=None), length=-1, id=None),
                    "xpos": Sequence(feature=ClassLabel(names=xpos, id=None), length=-1, id=None),
                    "tree_is_gold": Sequence(feature=Value(dtype="int16", id=None), length=-1, id=None),
                }
            )
            
            new_dataset[split] = Dataset.from_list(
                base_rows + treebank_rows,
                features=features,
            )
            self.logger.info(f"Appended split {split} with {len(new_dataset[split])} sequences")

        return datasets.DatasetDict(new_dataset).with_format("torch")
