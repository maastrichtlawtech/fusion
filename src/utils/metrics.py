import numpy as np
from statistics import mean
from collections import defaultdict


def compute_precision_recall_f1(gold: list[int], predicted: list[int]) -> dict:
    """
    Compute precision, recall and F1 score for a given query.

    :param gold: A list of relevant document ids.
    :param predicted: A list of retrieved document ids.
    :returns: A dictionary containing the computed metrics.
    """
    if predicted is None:
        return {'precision': 0, 'recall': 0, 'f1': 0}
    tp = len(set(gold) & set(predicted))
    fp = len(predicted) - tp
    fn = len(gold) - tp
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return {'precision': precision, 'recall': recall, 'f1': f1}


class Metrics:
    """
    Class to compute evaluation metrics for retrieval tasks.

    :param recall_at_k: A list of integers for recall@k.
    :param map_at_k: A list of integers for map@k.
    :param mrr_at_k: A list of integers for mrr@k.
    :param ndcg_at_k: A list of integers for ndcg@k.
    """
    def __init__(self, recall_at_k: list[int], map_at_k: list[int] = [], mrr_at_k: list[int] = [], ndcg_at_k: list[int] = []):
        self.recall_at_k = recall_at_k
        self.map_at_k = map_at_k
        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k

    def compute_all_metrics(self, all_ground_truths: list[list[int]], all_results: list[list[int]]) -> dict:
        """ 
        Compute all class metrics for a list of ground truths and results.

        :param all_ground_truths: A list of lists containing the ground truth document ids.
        :param all_results: A list of lists containing the retrieved document ids.
        :returns: A dictionary containing the computed metrics.
        """
        scores = defaultdict(dict)
        for k in self.recall_at_k:
            scores[f'recall@{k}'] = self.compute_mean_score(self.recall, all_ground_truths, all_results, k)
        for k in self.map_at_k:
            scores[f'map@{k}'] = self.compute_mean_score(self.average_precision, all_ground_truths, all_results, k)
        for k in self.mrr_at_k:
            scores[f'mrr@{k}'] = self.compute_mean_score(self.reciprocal_rank, all_ground_truths, all_results, k)
        for k in self.ndcg_at_k:
            scores[f'ndcg@{k}'] = self.compute_mean_score(self.ndcg, all_ground_truths, all_results, k)
        scores['r-precision'] = self.compute_mean_score(self.r_precision, all_ground_truths, all_results)
        return scores

    def compute_mean_score(self, score_func, all_ground_truths: list[list[int]], all_results: list[list[int]],  k: int = None):
        """
        Compute the mean score for a given metric.

        :param score_func: The metric function to use.
        :param all_ground_truths: A list of lists containing the ground truths.
        :param all_results: A list of lists containing the results.
        :param k: The value of k for the metric@k.
        :returns: The mean score for the metric.
        """
        return mean([score_func(truths, res, k) for truths, res in zip(all_ground_truths, all_results)])

    def average_precision(self, ground_truths: list[int], results: list[int], k: int = None):
        """
        Compute the average precision for a given query.

        :param ground_truths: A list of relevant document ids.
        :param results: A list of retrieved document ids.
        :param k: The value of k for the metric@k.
        :returns: The average precision for the query.
        """
        k = len(results) if k is None else k
        p_at_k = [self.precision(ground_truths, results, k=i+1) if d in ground_truths else 0 for i, d in enumerate(results[:k])]
        return sum(p_at_k)/len(ground_truths)

    def reciprocal_rank(self, ground_truths: list[int], results: list[int], k: int = None):
        """
        Compute the reciprocal rank for a given query.

        :param ground_truths: A list of relevant document ids.
        :param results: A list of retrieved document ids.
        :param k: The value of k for the metric@k.
        :returns: The reciprocal rank for the query.
        """
        k = len(results) if k is None else k
        return max([1/(i+1) if d in ground_truths else 0.0 for i, d in enumerate(results[:k])])

    def ndcg(self, ground_truths: list[int], results: list[int], k: int = None):
        """
        Compute the normalized discounted cumulative gain for a given query.

        :param ground_truths: A list of relevant document ids.
        :param results: A list of retrieved document ids.
        :param k: The value of k for the metric@k.
        :returns: The normalized discounted cumulative gain for the query.
        """
        k = len(results) if k is None else k
        relevances = [1 if d in ground_truths else 0 for d in results[:k]]
        dcg = relevances[0] + sum(relevances[i] / np.log2(i + 1) for i in range(1, len(relevances)))
        idcg = 1 + sum(1 / np.log2(i + 1) for i in range(1, len(ground_truths)))
        return (dcg / idcg) if idcg != 0 else 0

    def r_precision(self, ground_truths: list[int], results: list[int], R: int = None):
        """
        Compute the R-precision for a given query.

        :param ground_truths: A list of relevant document ids.
        :param results: A list of retrieved document ids.
        :param R: The value of R for the metric@R.
        :returns: The R-precision for the query.
        """
        R = len(ground_truths)
        relevances = [1 if d in ground_truths else 0 for d in results[:R]]
        return sum(relevances)/R

    def recall(self, ground_truths: list[int], results: list[int], k: int = None):
        """
        Compute the recall for a given query.

        :param ground_truths: A list of relevant document ids.
        :param results: A list of retrieved document ids.
        :param k: The value of k for the metric@k.
        :returns: The recall for the query.
        """
        k = len(results) if k is None else k
        relevances = [1 if d in ground_truths else 0 for d in results[:k]]
        return sum(relevances)/len(ground_truths)

    def precision(self, ground_truths: list[int], results: list[int], k: int = None):
        """
        Compute the precision for a given query.

        :param ground_truths: A list of relevant document ids.
        :param results: A list of retrieved document ids.
        :param k: The value of k for the metric@k.
        :returns: The precision for the query.
        """
        k = len(results) if k is None else k
        relevances = [1 if d in ground_truths else 0 for d in results[:k]]
        return sum(relevances)/len(results[:k])

    def fscore(self, ground_truths: list[int], results: list[int], k: int = None):
        """
        Compute the F-score for a given query.

        :param ground_truths: A list of relevant document ids.
        :param results: A list of retrieved document ids.
        :param k: The value of k for the metric@k.
        :returns: The F-score for the query.
        """
        p = self.precision(ground_truths, results, k)
        r = self.recall(ground_truths, results, k)
        return (2*p*r)/(p+r) if (p != 0.0 or r != 0.0) else 0.0
