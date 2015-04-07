#!/usr/bin/env python
import nltk
import pandas as pd
import numpy as np
import re
from math import cos, pi
from gensim.models import Word2Vec
from afinn_extender import load_afinn_file
import csv
import gzip
import argparse

# avoid 'broken pipe' error
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE, SIG_DFL)


''' Global variables
'''

score_regex = re.compile(r'.*(\d+)\s*(?:/|out of|out|of)\s*((?:100|10|5)).*')
html_regex = re.compile(r'<[^>]+>')
words_regex = re.compile(r'\w+')
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stopwords = nltk.corpus.stopwords.words('english')
w2v = Word2Vec.load('data/model.w2v')
w2v_vocab = w2v.vocab.iterkeys()
afinn = load_afinn_file('data/AFINN-111_extended.txt')


''' Some basic functions to be used below
'''

def normalize_dict(d):
    total = 0.
    for x in d.itervalues():
        total += x
    return {k: v/total for k, v in d.iteritems()}


def mean(l):
    return sum(l) / float(len(l)) if l else 0.


def remove_stopwords(sentence):
    words = words_regex.findall(sentence.lower())
    words = [w for w in words if w not in stopwords]
    return ' '.join(words)


def get_sentences(review):
    review = unicode(review, encoding='utf-8')
    review = html_regex.sub(' ', review)
    sentences = sentence_tokenizer.tokenize(review)
    return map(remove_stopwords, sentences)


''' Preprocessing functions needed to compute a weighted score based on AFINN word scores
'''

def get_weighted_sentences(sentences):
    n = len(sentences)
    if n > 1:
        weights = [2 + cos(2*pi*x / (n-1)) for x in xrange(n)]
        total = sum(weights)
        weights = [w/total for w in weights]

        return dict(zip(sentences, weights))
    elif n == 1:
        return {sentences[0]: 1.}
    else:
        return {}


def get_word_score(word):
    ''' This function throws an error if it is passed a word that
        is not in w2v_vocab of afinn.
        I chose not to check for this for performance reasons.
    '''
    if word in afinn:
        return float(afinn[word])
    
    similar = w2v.most_similar(word, topn=2500)
    similar = [(word, sim) for word, sim in similar if word in afinn]
    similar = sorted(similar, key=lambda tup: tup[1], reverse=True)
    similar = dict(similar[:10])
    similar = normalize_dict(similar)
    
    score = 0.
    for word, weight in similar.iteritems():
        score += weight*afinn[word]

    return score


def get_sentence_score(sentence):
    words = [w for w in sentence.split(' ') if w in w2v_vocab or w in afinn]
    return mean(map(get_word_score, words))


def get_afinn_score(review):
    weighted_sentences = get_weighted_sentences(get_sentences(review))
    score = 0.
    for sentence, weight in weighted_sentences.iteritems():
        score += weight*get_sentence_score(sentence)
    
    return score


''' Try to parse a score given explicitly in review
'''

def get_parsed_score(review):
    m = score_regex.match(review)
    if m:
        score = float(m.group(1)) / float(m.group(2))
        return 10.*score - 5.
    else:
        return 0.


''' Combine both methods
'''

def get_vw_format(row):

    # read input row
    row_id = row['id']
    label = 2*int(row.get('sentiment', '1')) - 1
    review = row['review']
    sentences = get_sentences(review)
    n = len(sentences)
    n_mean = 12.45

    # review sentences
    first_sentence = sentences[0]
    if n >= 3:
        last_two_sentences = ' '.join(sentences[-2:])
    elif n == 2:
        last_two_sentences = sentences[-1]
    else:
        last_two_sentences = ''

    # score directly parsed from review
    parsed_score = get_parsed_score(review)

    # score based on AFINN scores (via word2vec)
    afinn_score = get_afinn_score(review)

    # and their respective weights
    sentences_weight = n_mean / n
    first_sentence_weight = 1.
    last_two_sentences_weight = 1.
    parsed_score_weight = 2. if (parsed_score != 0.) else 0.
    afinn_score_weight = 1.

    # the output string
    s = "%d '%s" % (label, row_id)
    s += ' |p:%f %f' % (parsed_score_weight, parsed_score)
    s += ' |a:%f %f' % (afinn_score_weight, afinn_score)
    s += ' |s:%f %s' % (sentences_weight, sentences)
    s += ' |f:%f %s' % (first_sentence_weight, first_sentence)
    s += ' |l:%f %s' % (last_two_sentences_weight, last_two_sentences)
    
    return s


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Extract features and output in Vowpal Wabbit format',
        epilog='Example usage: python preprocess.py input.tsv.gz output.vw')

    parser.add_argument(
        'input_file',
        action='store',
        help='schema: {id: str, sentiment: {0,1}, review: str}, header: yes, delimiter: tab, compression: gzip')

    parser.add_argument(
        'output_file',
        action='store',
        help='plain text file in vw format')

    args = parser.parse_args()

    with gzip.open(args.input_file) as in_file, \
            open(args.output_file, 'w') as out_file:

        rows = csv.DictReader(in_file, delimiter='\t', escapechar='\\')
        for row in rows:
            out_file.write('%s\n' % get_vw_format(row))



