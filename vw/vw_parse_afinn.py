#!/usr/bin/env python
""" This code parses the movie review data and writes the rows to stdout in a
    format that Vowpal Wabbit understands.
"""

import re
import csv
import sys
import gzip
from math import sqrt

# avoid 'broken pipe' error
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE, SIG_DFL)

# average number of words in review
n_words = 243

# pre-compiled regex instances
re_html_tags = re.compile(r'<[^>]+>')
re_word_chars = re.compile(r'\w+', flags=(re.UNICODE|re.LOCALE))


# get mean and standard deviation of a list
def mu_sigma(a):
    mu = sum(a) / float(len(a)) if a else 0.
    sigma = sqrt(sum([(x - mu)**2 for x in a]) / len(a)) if a else 0.
    return mu, sigma


# this parses one line in AFINN word-score text file
def afinn_parse(row):
    word, score = row.strip().split('\t')
    return word, int(score)


# get the AFINN word-score list
with open('AFINN/AFINN-111.txt') as rows:
    afinn = dict(map(afinn_parse, rows))


# cleans a string "I'm a <br/>string!?" returns as "i m a string"
def parse(s):
    s = re_html_tags.sub(' ', s).strip()    # remove html tags
    s = s.lower()                           # lowercase
    s = re_word_chars.findall(s)            # pick out only word characters
    w = n_words / float(len(s))             # weigh on sentence length
    a = [afinn[word] for word in set(s).intersection(afinn)]  # afinn scores
    mu, sigma = mu_sigma(a)                 # mean and standard deviation
    s = ' '.join(s)                         # join into a sentence again
    return s, w, mu, sigma


# take a line and convert it to vw input format, see
# https://github.com/JohnLangford/vowpal_wabbit/wiki/Input-format
def vw_row(row):
    row_id = row['id']
    label = '-1' if (row.get('sentiment') == '0') else '1'
    review, w, mu, sigma = parse(row['review'])
    return "%s '%s |r:%f %s |a:1.0 mu:%f sigma:%f" % (label, row_id, w, review, mu, sigma)


# run
if __name__ == '__main__':

    # parse arguments
    args = sys.argv[1:]

    if args:
        path = args[0]
    else:
        raise Exception('please specify path to data file')

    # parse rows and write to stdout
    with gzip.open(path) as f:
        rows = csv.DictReader(f, delimiter='\t')
        for row in rows:
            print vw_row(row)

