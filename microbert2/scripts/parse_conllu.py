import math
import os
import sys

import more_itertools as mit
import stanza
from stanza.utils.conll import CoNLL
from tqdm import tqdm


def main():
    config = {
        "processors": "tokenize,pos,lemma,depparse",
        "lang": "en",
        "use_gpu": True,
        "logging_level": "INFO",
        "tokenize_pretokenized": False,
        "tokenize_no_ssplit": False,
    }
    pipeline = stanza.Pipeline(**config)

    text_file = sys.argv[1]
    output_conllu_file = sys.argv[2]
    batch_size = 128

    with open(text_file, "r") as fin, open(output_conllu_file, "w") as fout:
        chunks = list(mit.chunked(fin, batch_size))
        for chunk in tqdm(chunks):
            doc = pipeline("".join(chunk))
            fout.write(CoNLL.doc2conll_text(doc))


if __name__ == "__main__":
    main()
