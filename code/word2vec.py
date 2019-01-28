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
    model = gensim.models.Word2Vec(sentences, size=200, window=5, min_count=10, workers=4)
    model.save(model_file)


print('Pre-training word embeddings ...')
main('restaurant')
main('beer')
