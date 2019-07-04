import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils import get_predictor


class JigsawMetrics:
    @staticmethod
    def compute_auc(y_true, y_pred):
        try:
            return roc_auc_score(y_true, y_pred)
        except ValueError:
            return np.nan

    @staticmethod
    def compute_subgroup_auc(df, subgroup, label, pred):
        subgroup_examples = df[df[subgroup]]
        return JigsawMetrics.compute_auc(subgroup_examples[label], subgroup_examples[pred])

    @staticmethod
    def compute_bpsn_auc(df, subgroup, label, pred):
        """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
        subgroup_negative_examples = df[df[subgroup] & ~df[label]]
        non_subgroup_positive_examples = df[~df[subgroup] & df[label]]
        examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
        return JigsawMetrics.compute_auc(examples[label], examples[pred])

    @staticmethod
    def compute_bnsp_auc(df, subgroup, label, pred):
        """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
        subgroup_positive_examples = df[df[subgroup] & df[label]]
        non_subgroup_negative_examples = df[~df[subgroup] & ~df[label]]
        examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
        return JigsawMetrics.compute_auc(examples[label], examples[pred])

    @staticmethod
    def compute_bias_metrics_for_model(dataset, subgroups, label_col, pred_col):
        """Computes per-subgroup metrics for all subgroups and one model."""
        records = []
        for subgroup in subgroups:
            record = {'subgroup': subgroup, 'subgroup_size': len(dataset[dataset[subgroup]])}
            record['subgroup_auc'] = JigsawMetrics.compute_subgroup_auc(dataset, subgroup, label_col, pred_col)
            record['bpsn_auc'] = JigsawMetrics.compute_bpsn_auc(dataset, subgroup, label_col, pred_col)
            record['bnsp_auc'] = JigsawMetrics.compute_bnsp_auc(dataset, subgroup, label_col, pred_col)
            records.append(record)

        return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)

    @staticmethod
    def calculate_overall_auc(df, label_col, pred_col):
        true_labels = df[label_col]
        predicted_labels = df[pred_col]
        return JigsawMetrics.compute_auc(true_labels, predicted_labels)

    @staticmethod
    def power_mean(series, p):
        total = sum(np.power(series, p))
        return np.power(total / len(series), 1 / p)

    @staticmethod
    def get_final_metric(bias_df, overall_auc):
        power, weight = -5, 0.25
        bias_score = np.average([JigsawMetrics.power_mean(bias_df['subgroup_auc'], power),
                                 JigsawMetrics.power_mean(bias_df['bpsn_auc'], power),
                                 JigsawMetrics.power_mean(bias_df['bnsp_auc'], power)])
        return (weight * overall_auc) + ((1 - weight) * bias_score)

    def __call__(self, dataset, predictions):
        label_col = 'target'
        pred_col = 'model_prediction'
        identity_columns = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
                            'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

        for col in identity_columns + [label_col]:
            dataset[col] = dataset[col] >= 0.5
        dataset[pred_col] = predictions

        bias_metrics = JigsawMetrics.compute_bias_metrics_for_model(dataset, identity_columns, label_col, pred_col)
        overall_auc = JigsawMetrics.calculate_overall_auc(dataset, label_col, pred_col)
        final_metric = JigsawMetrics.get_final_metric(bias_metrics, overall_auc)

        return final_metric, bias_metrics


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='datasets/validation.csv', help="Path of data")
    parser.add_argument("--config_path", type=str, default='config.json', help="Path of config")
    parser.add_argument("--batch_size", type=int, default=32, help="")

    return parser


def main(args):
    def chunker(seq, size):
        return [seq[pos:pos + size] for pos in range(0, len(seq), size)]

    with open(args.config_path, 'r') as json_file:
        config = json.load(json_file)

    metrics = JigsawMetrics()
    dataset = pd.read_csv(args.data_path)[:40000]
    predictor = get_predictor(config)

    predictions = []
    for batch in tqdm(chunker(dataset, args.batch_size)):
        batch_predictions = predictor(batch['comment_text'])
        predictions.append(batch_predictions)
    predictions = np.concatenate(predictions)

    final_score, metrics = metrics(dataset, predictions)

    print(f'Final score = {final_score}')
    print(metrics)


if __name__ == "__main__":
    arg_parser = get_parser()
    args = arg_parser.parse_known_args()[0]
    main(args)
