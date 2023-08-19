import argparse
import re
import string

import numpy as np
import openai
import util.utils as utils
from datasketch import MinHash
from nltk import ngrams
from scipy import spatial
from tqdm import tqdm

# OPENAI embeddings


def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]


cos_dist = spatial.distance.cosine


# MinHash embeddings
# See: https://github.com/Cerebras/modelzoo/blob/main/modelzoo/transformers/data_processing/slimpajama/dedup/to_hash.py


def get_features(s, width):
    # lower cased
    s = s.lower()
    # remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # remove consecutive spaces, newlines, tabs in the middle and in the beginning / end
    s = re.sub(r"\s+", " ", s.strip())
    return map(lambda x: "".join(x), ngrams(s, width))


def get_hash(text, width=6, num_perm=128):
    m = MinHash(num_perm)
    for x in get_features(text, width):
        m.update(x.encode("utf8"))
    return m


def hash_dist(m1, m2):
    return m1.jaccard(m2)


# Dataset


def get_text(x):
    if x["input"] == "":
        return x["instruction"] + " " + x["output"]
    else:
        return x["instruction"] + " " + x["input"] + " " + x["output"]

def merge_data(num_split=6):
    data = []
    for i in range(num_split):
        npy = np.load(f"alpaca_embeds_{i}.npy")
        data.append(npy)
    data = np.vstack(data)
    np.save("alpaca_embeds.npy", data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--nums", type=int, default=10000)
    parser.add_argument("--num_split", type=int, default=6)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    if args.merge:
        merge_data(args.num_split)
        exit()

    data = utils.jload("alpaca_data.json")
    index = args.index
    nums = args.nums
    data = data[nums * index : nums * (index + 1)]
    print(f"Processing {len(data)} examples from index {index * nums}")

    embeds = []
    for i in tqdm(range(len(data))):
        text = get_text(data[i])
        embed = get_embedding(text)
        embeds.append(np.array(embed))
    embeds = np.vstack(embeds)

    np.save(f"alpaca_embeds_{index}.npy", embeds)
