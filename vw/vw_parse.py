#!/usr/bin/env python
""" This code parses the movie review data and writes the rows to stdout in a
    format that Vowpal Wabbit understands.
"""

import re
import csv
import sys
import gzip



# avoid 'broken pipe' error
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE, SIG_DFL)

# pre-compiled regex instances
re_html_tags = re.compile(r'<[^>]+>')
re_word_chars = re.compile(r'\w+', flags=(re.UNICODE|re.LOCALE))


# cleans a string "I'm a <br/>string!?" returns as "i m a string"
def clean(s):
    s = re_html_tags.sub(' ', s).strip()    # remove html tags
    s = ' '.join(re_word_chars.findall(s))  # only word chars
    s = s.lower()                           # lowercase
    return s


# take a line and convert it to vw input format, see
# https://github.com/JohnLangford/vowpal_wabbit/wiki/Input-format
def vw_row(row):
    row_id = row['id']
    label = '-1' if (row.get('sentiment') == '0') else '1'
    review_text = clean(row['review'])
    return "%s '%s |r %s" % (label, row_id, review_text)


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

