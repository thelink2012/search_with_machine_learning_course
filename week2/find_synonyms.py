import sys
import fasttext

STOP_WORDS = set([
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is",
    "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there",
    "these", "they", "this", "to", "was", "will", "with"
])

def main(threshold: float):
    model = fasttext.load_model("title_model_final.bin")
    for line in sys.stdin:
        word = line.strip()
        synonyms = []
        for similarity, neighbour_word in model.get_nearest_neighbors(word, k=1000):
            if similarity < threshold:
                continue
            if neighbour_word in STOP_WORDS:
                sys.stderr.write(f"found stop word {neighbour_word}\n")
                continue
            synonyms.append(neighbour_word)
        if len(synonyms) > 0:
            print(",".join([word, *synonyms]))


if __name__ == "__main__":
    main(float(sys.argv[1]))

