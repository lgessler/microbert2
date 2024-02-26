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
                    sentences.extend([[t["form"] for t in s] for s in conllu.parse_incr(f)])
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

        train_dataset = datasets.Dataset.from_list(
            [{"tokens": s} for s in train_sentences],
            features=datasets.Features({"tokens": datasets.Sequence(datasets.Value(dtype="string"))}),
        )
        dev_dataset = datasets.Dataset.from_list(
            [{"tokens": s} for s in dev_sentences],
            features=datasets.Features({"tokens": datasets.Sequence(datasets.Value(dtype="string"))}),
        )

        self.logger.info(f"First train sentence: {train_dataset[0]}")
        self.logger.info(f"First dev sentence: {dev_dataset[0]}")

        return DatasetDict({"train": train_dataset, "dev": dev_dataset})


@Step.register("microbert2.nahuatl.data::finalize")
class Finalize(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT = DatasetsFormat()

    def run(
        self,
        dataset: IterableDatasetDict,
        treebank_dataset: Optional[DatasetDict] = None,
        unlabeled_per_labeled: int = 8,
    ) -> DatasetDict:
        dataset = dataset.remove_columns(["tokens"])
        print(dataset)

        features = datasets.Features(
            {
                "input_ids": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
                "attention_mask": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
                "token_type_ids": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
                "token_spans": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
                "head": Sequence(feature=Value(dtype="int16", id=None), length=-1, id=None),
                "deprel": Sequence(feature=ClassLabel(names=["NONE"], id=None), length=-1, id=None),
                "xpos": Sequence(feature=ClassLabel(names=["NONE"], id=None), length=-1, id=None),
                "tree_is_gold": Sequence(feature=Value(dtype="int16", id=None), length=-1, id=None),
            }
        )

        new_dataset = {}
        for split, rows in dataset.items():

            def get_rows(rows, tree_is_gold=(0,)):
                def inner():
                    def process_row(v):
                        # - 2 because of CLS and SEP tokens
                        num_whole_tokens = len(v["token_spans"]) // 2 - 2
                        return {
                            "input_ids": v["input_ids"],
                            "token_type_ids": v["token_type_ids"],
                            "attention_mask": v["attention_mask"],
                            "token_spans": v["token_spans"],
                            "head": [int(i) for i in v["head"]] if "head" in v else [0] * num_whole_tokens,
                            "deprel": v["deprel"] if "deprel" in v else [0] * num_whole_tokens,
                            "xpos": v["xpos"] if "xpos" in v else [0] * num_whole_tokens,
                            "tree_is_gold": tree_is_gold,
                        }

                    for v in Tqdm.tqdm(rows, desc=f"Constructing {split}..", total=len(rows)):
                        yield process_row(v)

                return inner

            base_rows = get_rows(rows)
            treebank_rows = ()
            new_dataset[split] = Dataset.from_generator(
                lambda: itertools.chain(base_rows if isinstance(base_rows, list) else base_rows(), treebank_rows),
                features=features,
            )
            self.logger.info(f"Appended split {split} with {len(new_dataset[split])} sequences")

        return datasets.DatasetDict(new_dataset).with_format("torch")
