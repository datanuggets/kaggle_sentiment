#!/usr/bin/env python

import pandas as pd
import argparse
from sklearn.metrics import roc_auc_score


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Generate submission file from VW prediction file',
        epilog='Example usage: python generate_submission.py predictions.txt')

    parser.add_argument(
        'prediction_file',
        action='store',
        help='schema: {Proba(y=1|x): float, id: str}, header: no, delimiter: space')

    parser.add_argument(
        'submission_file',
        action='store',
        help='path of submission file')

    args = parser.parse_args()
    pred = pd.read_csv(args.prediction_file, sep=' ', names=['sentiment', 'id'], index_col='id')
    pred.to_csv(args.submission_file)
