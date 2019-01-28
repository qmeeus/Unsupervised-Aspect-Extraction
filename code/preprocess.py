import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


def parseSentence(line):
    lmtzr = WordNetLemmatizer()    
    stop = stopwords.words('english')
    text_token = CountVectorizer().build_tokenizer()(line.lower())
    text_rmstop = [i for i in text_token if i not in stop]
    text_stem = [lmtzr.lemmatize(w) for w in text_rmstop]
    return text_stem


def preprocess_train(domain):
    train_file = '../datasets/' + domain + '/train.txt'
    output_dir = '../preprocessed_data/' + domain
    output_file = output_dir + '/train.txt'
    os.makedirs(output_dir, exist_ok=True)

    with open(train_file) as f, open(output_file, 'w') as out:
        for line in f:
            tokens = parseSentence(line)
            if len(tokens) > 0:
                out.write(' '.join(tokens)+'\n')

def preprocess_test(domain):
    # For restaurant domain, only keep sentences with single 
    # aspect label that in {Food, Staff, Ambience}
    output_dir = '../preprocessed_data/' + domain
    test_file = '../datasets/' + domain + '/test.txt'
    labels_file = '../datasets/' + domain + '/test_label.txt'
    output_file = output_dir + '/test.txt'
    output_labels = output_dir + '/test_label.txt'
    os.makedirs(output_dir, exist_ok=True)

    with open(test_file) as f1, open(labels_file) as f2, \
        open(output_file, 'w') as out1, open(output_labels, 'w') as out2:
        for text, label in zip(f1.readlines(), f2.readlines()):
            label = label.strip()
            if domain == 'restaurant' and label not in ['Food', 'Staff', 'Ambience']:
                continue
            tokens = parseSentence(text)
            if len(tokens) > 0:
                out1.write(' '.join(tokens) + '\n')
                out2.write(label+'\n')

def preprocess(domain):
    print('\t' + domain + ' train set ...')
    preprocess_train(domain)
    print('\t' + domain + ' test set ...')
    preprocess_test(domain)


def maybe_install_nltk_deps():
    import nltk
    for pkg in ("corpora/stopwords", "corpora/wordnet"):
        try:
            nltk.data.find(pkg)
        except LookupError:
            nltk.download(pkg.split("/")[-1])


print('Preprocessing raw review sentences ...')
maybe_install_nltk_deps()
preprocess('restaurant')
preprocess('beer')
