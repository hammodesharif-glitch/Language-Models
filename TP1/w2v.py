"""Static word embeddings with skip-gram with negative sampling (TP word2vec).

Usage: python w2v.py CORPUS OUTPUT [-n 100] [-L 2] [-k 10] [--eta 0.1] [-e 5] [--minc 5] [--seed S]

CORPUS is a segmented text: tokens separated by whitespace.
OUTPUT receives one embedding per line (TP section 5 format).
"""
import argparse
import sys
from collections import Counter

import numpy as np


# ---------------------------------------------------------------- corpus and vocabulary

def read_tokens(path):
    """Return the whitespace-separated tokens of a segmented text, in order."""
    with open(path, encoding="utf-8") as f:
        return f.read().split()


def build_vocab(tokens, minc):
    """Keep words occurring at least minc times.

    Returns (words, counts): words sorted by decreasing frequency (ties keep
    first-appearance order), counts[i] the number of occurrences of words[i].
    """
    counter = Counter(tokens)
    words = [w for w, c in counter.most_common() if c >= minc]
    counts = np.array([counter[w] for w in words], dtype=np.int64)
    return words, counts


def encode(tokens, word2idx):
    """Map tokens to vocabulary indices, dropping words below minc from the stream."""
    return np.array([word2idx[t] for t in tokens if t in word2idx], dtype=np.int64)


# ---------------------------------------------------------------- classifier

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def init_matrices(vocab_size, n, rng):
    """Random target (M) and context (C) embedding matrices, one row per word."""
    M = (rng.random((vocab_size, n)) - 0.5) / n
    C = (rng.random((vocab_size, n)) - 0.5) / n
    return M, C


def sgd_step(M, C, target, c_pos, c_negs, eta):
    """One mini-batch update: one positive example and its k negatives (TP eq. 3).

    Updates M[target], C[c_pos] and C[c_negs] in place; returns the loss (TP eq. 1)
    computed before the update.
    """
    ctx_ids = np.concatenate(([c_pos], c_negs))
    m = M[target]
    ctx = C[ctx_ids]                         # (k+1, n): c_pos then the k negatives
    scores = ctx @ m                         # m . c for each context word
    labels = np.zeros(len(ctx_ids))
    labels[0] = 1.0
    g = sigmoid(scores) - labels             # [sigma(m.c_pos) - 1], [sigma(m.c_neg_i)]

    grad_m = g @ ctx                         # dL/dm, with the context vectors at time t
    grad_ctx = np.outer(g, m)                # dL/dc for each row, with m at time t
    M[target] -= eta * grad_m
    np.subtract.at(C, ctx_ids, eta * grad_ctx)   # accumulates when an index repeats

    return np.logaddexp(0.0, -scores[0]) + np.logaddexp(0.0, scores[1:]).sum()


# ---------------------------------------------------------------- training

def train(ids, vocab_size, n, L, k, eta, epochs, rng):
    """Skip-gram with negative sampling over the encoded text; returns M."""
    M, C = init_matrices(vocab_size, n, rng)
    offsets = [j for j in range(-L, L + 1) if j != 0]
    for epoch in range(1, epochs + 1):
        total_loss, n_batches = 0.0, 0
        for i, target in enumerate(ids):
            for j in offsets:
                if not 0 <= i + j < len(ids):
                    continue
                c_negs = rng.integers(0, vocab_size, size=k)   # uniform over the lexicon (TP 2.1)
                total_loss += sgd_step(M, C, target, ids[i + j], c_negs, eta)
                n_batches += 1
        print(f"epoch {epoch}/{epochs}: mean loss {total_loss / max(n_batches, 1):.4f} "
              f"over {n_batches} mini-batches", file=sys.stderr)
    return M


# ---------------------------------------------------------------- output

def write_embeddings(path, words, M):
    """Write the TP section 5 format: a header "N d", then "word v1 ... vd" per line."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{M.shape[0]} {M.shape[1]}\n")
        for word, vec in zip(words, M):
            f.write(word + " " + " ".join(f"{x:.6f}" for x in vec) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("corpus", help="segmented text (whitespace-separated tokens)")
    parser.add_argument("output", help="embeddings file to write")
    parser.add_argument("-n", type=int, default=100, help="embedding dimension")
    parser.add_argument("-L", type=int, default=2, help="left and right context size (window 2L+1)")
    parser.add_argument("-k", type=int, default=10, help="negative examples per positive example")
    parser.add_argument("--eta", type=float, default=0.1, help="learning rate")
    parser.add_argument("-e", type=int, default=5, help="number of iterations over the corpus")
    parser.add_argument("--minc", type=int, default=5, help="minimum occurrences for a word to get an embedding")
    parser.add_argument("--seed", type=int, default=None, help="random seed (default: different each run)")
    args = parser.parse_args()

    tokens = read_tokens(args.corpus)
    words, counts = build_vocab(tokens, args.minc)
    if not words:
        sys.exit(f"no word occurs at least {args.minc} times in {args.corpus}")
    ids = encode(tokens, {w: i for i, w in enumerate(words)})
    print(f"{len(tokens)} tokens, {len(set(tokens))} distinct; kept {len(words)} words "
          f"(minc={args.minc}) covering {len(ids)} tokens", file=sys.stderr)

    rng = np.random.default_rng(args.seed)
    M = train(ids, len(words), args.n, args.L, args.k, args.eta, args.e, rng)
    write_embeddings(args.output, words, M)
    print(f"wrote {len(words)} embeddings of dimension {args.n} to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
