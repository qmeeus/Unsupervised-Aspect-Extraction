import gensim
import os


class MySentences(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        with open(self.filename) as f:
            for line in f.readlines():
                yield line.split()


def main(domain):
    source = '../datasets/%s/train.txt' % (domain)
    output_dir = "../preprocessed_data/%s" % (domain)
    os.makedirs(output_dir, exist_ok=True)
    model_file = output_dir + '/w2v_embedding'
    sentences = MySentences(source)
    model = gensim.models.Word2Vec(size=200, window=5, min_count=10, workers=4)
    model.build_vocab(sentences)  # TODO: is it necessary?
    model.save(model_file)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Train Word2Vec model")
    parser.add_argument("--dataset", choices=["restaurant", "beer"])
    args = parser.parse_args()
    print('Pre-training word embeddings ...')
    if not hasattr(args, "dataset") or args.dataset == "restaurant":
        main('restaurant')
    if not hasattr(args, "dataset") or args.dataset == "beer":
        main('beer')
