import math
import random
from itertools import chain, repeat

import torch
from tango import DillFormat, Step

from microbert2.microbert.tasks.task import MicroBERTTask


def ncycles(iterable, n):
    "Returns the sequence elements n times"
    return chain.from_iterable(repeat(tuple(iterable), n))


def rescale(base, target):
    target = int(target)
    if len(base) == target:
        return base
    elif len(base) > target:
        base = base.copy()
        random.shuffle(base)
        return base[:target]
    else:
        base = base.copy()
        scale = math.ceil(target / len(base))
        return (base * scale)[:target]


@Step.register("microbert2.data.combine::combine_datasets")
class CombineDatasets(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT = DillFormat()

    def run(
        self,
        datasets: list[dict],
        tasks: list[MicroBERTTask] = [],
    ) -> dict:

        base_inst = {}
        for task in tasks:
            for k in task.data_keys:
                base_inst[k] = task.null_tensor(k)

        def process_row(row, task, task_index):
            result = base_inst.copy()
            result["input_ids"] = torch.tensor(row["input_ids"])
            result["token_type_ids"] = torch.tensor(row["token_type_ids"])
            result["attention_mask"] = torch.tensor(row["attention_mask"])
            result["token_spans"] = torch.tensor(row["token_spans"])
            if task is None:
                result["dataset_id"] = torch.tensor(0)
            else:
                for k in task.data_keys:
                    result[k] = task.tensorify_data(k, row[k])
                result["dataset_id"] = torch.tensor(task_index)
            return result

        mlm_dataset = {}
        for split, rows in datasets[0].items():
            base_rows = [process_row(v, None, 0) for v in rows]
            mlm_dataset[split] = base_rows
            self.logger.info(f"Appended split {split} with {len(mlm_dataset[split])} sequences")

        i = 1
        task_datasets = []
        for task, dataset in zip(tasks, datasets[1:]):
            task_dataset = {}
            for split, insts in dataset.items():
                base_dataset = [process_row(v, task, i) for v in insts]
                if split == "train":
                    scaled_dataset = rescale(base_dataset, len(mlm_dataset["train"]) * task.inst_proportion)
                    self.logger.info(
                        f"Rescaled train split for {task.slug} from {len(base_dataset)} to {len(scaled_dataset)}"
                    )
                    task_dataset[split] = scaled_dataset
                else:
                    task_dataset[split] = base_dataset
            i += 1
            task_datasets.append(task_dataset)

        word_count = lambda insts: sum(len(inst["token_spans"]) - 2 for inst in insts)
        wp_count = lambda insts: sum(len(inst["input_ids"]) - 2 for inst in insts)
        for split, insts in mlm_dataset.items():
            self.logger.info(
                f"mlm_{split} size: {len(insts)} sentences, {word_count(insts)} words, {wp_count(insts)} wordpieces"
            )
        for i, ds in enumerate(task_datasets):
            task = tasks[i]
            for split, insts in ds.items():
                self.logger.info(
                    f"{task.slug}_{split} size: {len(insts)} sentences, {word_count(insts)} words, {wp_count(insts)} wordpieces"
                )

        for split in mlm_dataset.keys():
            for task_dataset in task_datasets:
                mlm_dataset[split] += task_dataset[split]

        return mlm_dataset
