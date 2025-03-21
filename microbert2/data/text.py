from typing import Optional

import more_itertools as mit
import stanza
from tango import DillFormat, Step
from tango.common import Tqdm


def initialize_stanza_pipeline(stanza_retokenize, stanza_use_mwt, stanza_language_code):
    if stanza_retokenize:
        config = {
            "processors": "tokenize,mwt" if stanza_use_mwt else "tokenize",
            "lang": stanza_language_code,
            "use_gpu": True,
            "logging_level": "INFO",
            "tokenize_pretokenized": False,
            "tokenize_no_ssplit": True,
        }
        return stanza.Pipeline(**config)


def retokenize(step, pipeline, stanza_use_mwt, sentences, path):
    batch_size = 256

    space_separated = [" ".join(ts) for ts in sentences]
    chunks = list(mit.chunked(space_separated, batch_size))

    outputs = []
    for chunk in Tqdm.tqdm(chunks, desc=f"Retokenizing {path} with Stanza..."):
        output = pipeline("\n\n".join(chunk))
        for s in output.sentences:
            retokenized = [t.text for t in (s.words if stanza_use_mwt else s.tokens)]
            if len(retokenized) != len(sentences[len(outputs)]):
                step.logger.info(f"Retokenized sentence:\n\tOld: {sentences[len(outputs)]}\n\tNew: {retokenized}\n\n")
            outputs.append(retokenized)
    return outputs


@Step.register("microbert2.data.text::read_whitespace_tokenized_text")
class ReadWhitespaceTokenizedText(Step):
    """
    Reads whitespace-tokenized text from a file and returns a dictionary with train, dev, and test keys.
    Each split is a list of instances, where each instance is a dictionary with a "tokens" key.
    Stanza may also be used to retokenize the text.
    """

    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT = DillFormat()

    def run(
        self,
        train_path: str,
        dev_path: str,
        test_path: str,
        stanza_retokenize: bool = False,
        stanza_use_mwt: bool = True,
        stanza_language_code: Optional[str] = None,
    ) -> dict:
        pipeline = initialize_stanza_pipeline(stanza_retokenize, stanza_use_mwt, stanza_language_code)

        def read_from_path(filepath):
            sentences = []
            token_count = 0
            with open(filepath, "r") as f:
                for x in f:
                    x = x.strip()
                    x = [x for x in x.split(" ") if x != ""]
                    token_count += len(x)
                    sentences.append(x)
            return sentences

        train_sentences = read_from_path(train_path)
        dev_sentences = read_from_path(dev_path)
        test_sentences = read_from_path(test_path)
        if stanza_retokenize:
            train_sentences = retokenize(self, pipeline, stanza_use_mwt, train_sentences, train_path)
            dev_sentences = retokenize(self, pipeline, stanza_use_mwt, dev_sentences, dev_path)
            test_sentences = retokenize(self, pipeline, stanza_use_mwt, test_sentences, test_path)

        self.logger.info(
            f"Read {len(train_sentences)} train_sentences "
            f"({sum(len(x) for x in train_sentences)} tokens) from {train_path}"
        )
        self.logger.info(
            f"Read {len(dev_sentences)} dev_sentences " f"({sum(len(x) for x in dev_sentences)} tokens) from {dev_path}"
        )
        self.logger.info(
            f"Read {len(test_sentences)} test_sentences "
            f"({sum(len(x) for x in test_sentences)} tokens) from {test_path}"
        )
        train_dataset = [{"tokens": s} for s in train_sentences]
        dev_dataset = [{"tokens": s} for s in dev_sentences]
        test_dataset = [{"tokens": s} for s in test_sentences]
        self.logger.info(f"First train sentence: {train_dataset[0]}")
        self.logger.info(f"First dev sentence: {dev_dataset[0]}")
        self.logger.info(f"First test sentence: {test_dataset[0]}")

        return {"train": train_dataset, "dev": dev_dataset, "test": test_dataset}
