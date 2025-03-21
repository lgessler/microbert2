import math
import random
from itertools import chain, repeat

import torch
from tango import DillFormat, Step

from microbert2.microbert.tasks.mlm import MLMTask
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
        assert isinstance(tasks[0], MLMTask), "First task must be MLMTask"

        def process_row(row, task, task_index):
            result = base_inst.copy()
            result["input_ids"] = torch.tensor(row["input_ids"])
            result["token_type_ids"] = torch.tensor(row["token_type_ids"])
            result["attention_mask"] = torch.tensor(row["attention_mask"])
            result["token_spans"] = torch.tensor(row["token_spans"])
            result["dataset_id"] = torch.tensor(task_index)

            if task is not None:
                for k in task.data_keys:
                    # MLM labels are generated dynamically, so skip
                    if k == "labels" and task.slug == "mlm":
                        continue
                    result[k] = task.tensorify_data(k, row[k])
            return result

        # Process all datasets including MLM
        task_datasets = []
        for i, (task, dataset) in enumerate(zip(tasks, datasets)):
            task_dataset = {}
            for split, insts in dataset.items():
                if split == "train":
                    self.logger.info(f"\n\nFirst train instance for {task.slug}: {insts[0]}")
                base_dataset = [process_row(v, task, i) for v in insts]
                if split == "train" and i > 0:  # Skip scaling for MLM task (i=0)
                    # Scale based on MLM dataset size
                    mlm_size = len(datasets[0]["train"])
                    scaled_dataset = rescale(base_dataset, mlm_size * task.inst_proportion)
                    self.logger.info(
                        f"Rescaled train split for {task.slug} from {len(base_dataset)} to {len(scaled_dataset)}"
                    )
                    task_dataset[split] = scaled_dataset
                else:
                    task_dataset[split] = base_dataset
            task_datasets.append(task_dataset)

        # Log statistics for each task
        word_count = lambda insts: sum(len(inst["token_spans"]) - 2 for inst in insts)
        wp_count = lambda insts: sum(len(inst["input_ids"]) - 2 for inst in insts)

        for i, (task, ds) in enumerate(zip(tasks, task_datasets)):
            for split, insts in ds.items():
                self.logger.info(
                    f"{task.slug}_{split} size: {len(insts)} sentences, {word_count(insts)} words, {wp_count(insts)} wordpieces"
                )

        # Combine all datasets
        combined_dataset = {}
        for split in task_datasets[0].keys():
            combined_dataset[split] = []
            for task_dataset in task_datasets:
                combined_dataset[split] += task_dataset[split]

        return combined_dataset
