"""Transformer le corpus BPE en corpus au niveau des mots pour w2v.py.

Utilisation : python prepare_corpus.py INPUT.bpe OUTPUT.tok [--max-words N]
"""
import argparse


def bpe_to_words(tokens):
    """Recoller les morceaux '#' au mot précédent et supprimer <s> / </s>."""
    words = []
    for tok in tokens:

        if tok == "<s>" or tok == "</s>":
            continue

        if tok.startswith("#"):
            words[-1] = words[-1] + tok[1:]
        else:
            words.append(tok)

    return words


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument(
        "--max-words",
        type=int,
        default=None,
        help="garder seulement les N premiers mots (pour des expériences plus rapides)"
    )
    args = parser.parse_args()

    tokens = []

    for line in open(args.input, encoding="utf-8"):
        tokens.extend(line.split())

    words = bpe_to_words(tokens)
    if args.max_words is not None:
        words = words[:args.max_words]

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(" ".join(words))

    print(f"Nombre de mots écrits : {len(words)}")


if __name__ == "__main__":
    main()