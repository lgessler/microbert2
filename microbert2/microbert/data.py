from itertools import chain, repeat

import torch
from tango import DillFormat, Step


def ncycles(iterable, n):
    "Returns the sequence elements n times"
    return chain.from_iterable(repeat(tuple(iterable), n))


@Step.register("microbert2.microbert.data::combine_datasets")
class CombineDatasets(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT = DillFormat()

    # Note: we are expecting "full conllu" in the dataset argument here, as would be produced by
    # microbert2.data.stanza::stanza_parse_dataset
    def run(
        self,
        dataset: dict,
    ) -> dict:
        def process_row(v):
            # - 2 because of CLS and SEP tokens
            num_whole_tokens = len(v["token_spans"]) // 2 - 2
            return {
                "input_ids": torch.tensor(v["input_ids"]),
                "token_type_ids": torch.tensor(v["token_type_ids"]),
                "attention_mask": torch.tensor(v["attention_mask"]),
                "token_spans": torch.tensor(v["token_spans"]),
            }

        new_dataset = {}
        for split, rows in dataset.items():
            base_rows = [process_row(v) for v in rows]
            new_dataset[split] = base_rows
            self.logger.info(f"Appended split {split} with {len(new_dataset[split])} sequences")

        return new_dataset
