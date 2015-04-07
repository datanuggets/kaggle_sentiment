#!/usr/bin/env python

import pandas as pd
import argparse
from sklearn.metrics import roc_auc_score
from math import cos, pi


def cos_profile(x, n=1):
    x = (1-cos(pi*x)) / 2
    while n > 1:
        x = (1-cos(pi*x)) / 2
        n -= 1
    return x


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Get ROC AUC score of prediction file output by e.g. Vowpal Wabbit',
        epilog='Example usage: python validate.py predictions.txt validation.tsv.gz')

    parser.add_argument(
        'prediction_file',
        action='store',
        help='schema: {Proba(y=1|x): float, id: str}, header: no, delimiter: space, compression: none')

    parser.add_argument(
        'validation_file',
        action='store',
        help='schema: {id: str, sentiment: {0,1}, review: str}, header: yes, delimiter: tab, compression: gzip')

    parser.add_argument(
        '-p',
        '--profile',
        choices=['linear', 'step', 'cos'],
        action='store',
        default='linear',
        help='what kind profile function to apply to the output p = Proba(y=1|x)')

    parser.add_argument(
        '-n',
        '--nest',
        choices=range(1, 10),
        type=int,
        action='store',
        default=1,
        help='number of nested applications of cosine profile function')

    args = parser.parse_args()

    validation = pd.read_csv(args.validation_file, sep='\t', index_col='id', compression='gzip')
    prediction = pd.read_csv(args.prediction_file, sep=' ', index_col='id',
                             names=['sentiment_proba', 'id'])

    prediction = prediction.join(validation)

    y_true = prediction.sentiment.values
    if args.profile == 'linear':
        y_pred = prediction.sentiment_proba.values
    elif args.profile == 'step':
        y_pred = prediction.sentiment_proba.map(round).values
    elif args.profile == 'cos':
        y_pred = prediction.sentiment_proba.map(lambda p: cos_profile(p, args.nest)).values


    print 'ROC AUC score: %f' % roc_auc_score(y_true, y_pred)
