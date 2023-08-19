import argparse

import numpy as np
import util.utils as utils
from dq.methods.methods_utils.submodular_function import GraphCut
from dq.methods.methods_utils.submodular_optimizer import NaiveGreedy


def random_sample(data, n=1000):
    indices = np.random.choice(len(data), n, replace=False)
    return [data[i] for i in indices]


def dataset_quantization(data, ratio=0.02, k=50):
    n = int(len(data) * ratio)
    bins_n = int(n / k)
    budget_n = len(data) // k
    print(f"total: {len(data)} n: {n}, k: {k}, budget_n: {budget_n}, bins_n: {bins_n}")

    embeddings_original = np.load("alpaca_embeds.npy")
    embeddings = embeddings_original.copy()
    indices_original = np.arange(len(data))
    indices = indices_original.copy()

    sim_matrix = lambda a, b: embeddings[a] @ embeddings[b].T

    # bin generation
    bins = []
    for i in range(k):
        print(f"bin {i}/{k}")
        submod_f = GraphCut(index=indices, similarity_kernel=sim_matrix)
        submod_opt = NaiveGreedy(args=None, index=indices, budget=budget_n)
        result_indices = submod_opt.select(
            gain_function=submod_f.calc_gain,
            update_state=submod_f.update_state,
        )

        bins.append(result_indices)
        indices = np.delete(indices_original, np.concatenate(bins))
        embeddings = np.delete(embeddings_original, np.concatenate(bins), axis=0)

    # bin sampling
    index = []
    assert len(bins) == k
    for i in range(k):
        sampled_indices = random_sample(bins[i], n=bins_n)
        index.extend(sampled_indices)
    data = [data[i] for i in index]
    print(f"sampled: {len(data)} examples")

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    data = utils.jload("alpaca_data.json")
    total_len = len(data)

    # random sample
    if args.random:
        n = int(total_len * args.ratio)
        data = random_sample(data, n=n)
        print(f"Random sample: {len(data)} examples")
        utils.jdump(data, f"alpaca_data_random.json")
    # DQ
    else:
        k = args.k
        ratio = args.ratio
        data = dataset_quantization(data, ratio=ratio, k=k)
        print(f"DQ: {len(data)} examples")
        utils.jdump(data, f"alpaca_data_dq_k{k}.json")
