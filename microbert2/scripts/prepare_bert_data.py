import math
import os

from datasets import load_dataset


def main():
    wikipedia = load_dataset("wikipedia", "20220301.en")
    bookcorpus = load_dataset("bookcorpus")

    os.makedirs("data/bert/train")
    os.makedirs("data/bert/dev")

    train1 = wikipedia["train"]["text"]
    train2 = bookcorpus["train"]["text"]
    dev = wikitext["validation"]["text"]

    suitable = lambda s: len(s.strip()) > 0

    train1 = [x.strip() for x in train1 if suitable(x)]
    train2 = [x.strip() for x in train2 if suitable(x)]
    # Downsample a bit so we have around 2.5B tokens
    print(sum(len(s.split(" ")) for s in train1))
    print(sum(len(s.split(" ")) for s in train2))
    dev = [x.strip() for x in dev if suitable(x)]

    def write_bare_text(path, xs):
        with open(path, "w") as f:
            f.write("\n".join(xs) + "\n")

    write_bare_text("data/bert/train/train.txt", train1 + train2)
    write_bare_text("data/bert/dev/dev.txt", dev)


if __name__ == "__main__":
    main()
