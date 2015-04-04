import argparse
import nltk
import pandas
import re

from bs4 import BeautifulSoup
from gensim.models import Word2Vec


def load_afinn_file(path):
    def afinn_parse(row):
        word, score = row.strip().split('\t')
        return word, int(score)
    with open(path) as rows:
        return dict(map(afinn_parse, rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extends an AFINN words list with similar words as given by a Word2Vec model.'
    )
    parser.add_argument(
        'afinn_input',
        action='store',
        help='AFINN file to extend'
    )
    parser.add_argument(
        'afinn_output',
        action='store',
        help='AFINN file to create'
    )
    group = parser.add_mutually_exclusive_group(
        required=True
    )
    group.add_argument(
        '--corpus_file',
        action='store'
    )
    group.add_argument(
        '--word2vec_model',
        action='store'
    )
    parser.add_argument(
        '--strip_stopwords',
        action='store_true',
        help='Remove stopwords from training corpus'
    )

    parser.add_argument(
        '--save_word2vec',
        action='store',
        help='Save the created Word2Vec model'
    )
    args = parser.parse_args()


    if args.corpus_file:
        words_regex = re.compile('\w+', re.UNICODE)
        sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        stopwords = set(nltk.corpus.stopwords.words('english')) if args.strip_stopwords else {}

        print "Pre-processing unlabled data..."
        sentences = []
        corpus = pandas.read_csv(args.corpus_file, header=0,  delimiter="\t", quoting=3)
        for review in corpus['review']:
          review = BeautifulSoup(review).get_text()
          for line in sentence_tokenizer.tokenize(review):
              line = line.lower()
              line = [w for w in words_regex.findall(line) if w not in stopwords]
              sentences.append(line)

        print "Training Word2Vec model..."
        w2v = Word2Vec(
          sentences,
          workers=4,  # Number of threads to run in parallel
            size=300,  # Word vector dimensionality
            min_count=40,  # Minimum word count
            window=10,  # Context window size
            sample=1e-3,  # Downsample setting for frequent words
            seed=1
        )

        w2v.init_sims(replace=True)  # Makes the model much more memory-efficient

        if args.save_word2vec:
            w2v.save(args.save_word2vec)

    elif args.word2vec_model:
        w2v = Word2Vec.load(args.word2vec_model)
    else:
        raise Exception("Unsupported settings")


    print "Getting extra AFINN words..."

    afinn = load_afinn_file(args.afinn_input)
    positive_afinn_words = [w for w, s in afinn.iteritems() if w in w2v and w >= 2]
    negative_afinn_words = [w for w, s in afinn.iteritems() if w in w2v and w <= -2]

    with open(args.afinn_output, 'wb') as f:
        # Write original words
        for word, sentiment in afinn.iteritems():
            f.write("%s\t%d\n" % (word, sentiment))
        # Write newly found positive words
        for word, similarity in w2v.most_similar(positive_afinn_words, negative_afinn_words, topn=75):
            f.write("%s\t%d\n" % (word, 2))
        # Write newly found negative words
        for word, similarity in w2v.most_similar(negative_afinn_words, positive_afinn_words, topn=75):
            f.write("%s\t%d\n" % (word, -2))

    print "Done"