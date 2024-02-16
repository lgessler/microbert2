from typing import List

from tango import DillFormat, Step
from tango.common import DatasetDictBase


@Step.register("microbert2.data.util::count_unique_values")
class CountUniqueValues(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT = DillFormat()

    def run(self, dataset: DatasetDictBase, keys: List[str]):
        output = {}
        for k in keys:
            vals = set(dataset["train"].features[k].feature.names) | set(dataset["dev"].features[k].feature.names)
            output[k] = len(vals)
            self.logger.info(f"{len(vals)} unique values for key {k}")

        return output
