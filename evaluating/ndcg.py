"""
Create by Ken at 2020 May 12
Compute NDCG for Legal document retrieval task
Relevant score: 0, 1
Reference: https://towardsdatascience.com/evaluate-your-recommendation-engine-using-ndcg-759a851452d1
"""
import math


def dcg(predicted, ground_truth):
    n = len(predicted)
    score = 0
    for i in range(n):
        if predicted[i] in ground_truth:
            score += 1 / math.log(i + 2, 2)  # Add 2 because of 0-based index
    return score


def i_dcg(ground_truth):
    n = len(ground_truth)
    score = 0
    for i in range(n):
        score += 1 / math.log(i + 2, 2)  # Add 2 because of 0-based index
    return score


def n_dcg(predicted, ground_truth):
    return dcg(predicted, ground_truth) / i_dcg(ground_truth)
