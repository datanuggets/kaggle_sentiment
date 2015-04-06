#!/usr/bin/env python

import pandas as pd
import argparse
from sklearn.metrics import roc_auc_score


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

    args = parser.parse_args()

    validation = pd.read_csv(args.validation_file, sep='\t', index_col='id', compression='gzip')
    prediction = pd.read_csv(args.prediction_file, sep=' ', index_col='id',
                             names=['sentiment_proba', 'id'])

    prediction = prediction.join(validation)

    y_true = prediction.sentiment.values
    y_pred = prediction.sentiment_proba.values

    print 'ROC AUC score:', roc_auc_score(y_true, y_pred)
