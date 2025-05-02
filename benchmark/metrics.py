import json
import logging
from typing import List

logging.basicConfig(level=logging.ERROR)

import nltk
from rouge_score import rouge_scorer

from benchmark.llm_tools import GPTInterface, LLMInterface, OllamaInterface

def str_to_float(num_string: str) -> float:
    if num_string.endswith("%"):
        return float(num_string.strip("%")) / 100
    return float(num_string)

class Metric:
    name = "Metric"

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, predicted: List[str] | float | int | str, target: List[str] | float | int | str) -> float:
        raise NotImplementedError("Metric must implement __call__ method!")


class Precision(Metric):
    name = "precision"

    def __call__(self, predicted: List[str] | str, target: List[str] | str):
        if isinstance(predicted, str) and isinstance(target, str):
            rouge = rouge_scorer.RougeScorer(['rouge1'])
            results = rouge.score(target=target, prediction=predicted)
            return results['rouge1'].precision
        if isinstance(predicted, list) and isinstance(target, str):
            target = json.loads(target)
        if isinstance(predicted, list) and isinstance(target, list):
            normalize = lambda s: s.strip().lower()
            pred_set = set(map(normalize, predicted))
            target_set = set(map(normalize, target))
            if not pred_set:
                return 0.0
            true_positives = pred_set & target_set
            precision = len(true_positives) / len(pred_set)
            return precision
        logging.error("TypeError: Precision Metric: unsupported argument types")
        return 0.0


class Recall(Metric):
    name = "recall"

    def __call__(self, predicted: List[str] | str, target: List[str] | str):
        if isinstance(predicted, str) and isinstance(target, str):
            rouge = rouge_scorer.RougeScorer(['rouge1'])
            results = rouge.score(target=target, prediction=predicted)
            return results['rouge1'].recall
        if isinstance(predicted, list) and isinstance(target, str):
            target = json.loads(target)
        if isinstance(predicted, list) and isinstance(target, list):
            normalize = lambda s: s.strip().lower()
            pred_set = set(map(normalize, predicted))
            target_set = set(map(normalize, target))
            if not target_set:
                return 0.0
            true_positives = pred_set & target_set
            precision = len(true_positives) / len(target_set)
            return precision
        logging.error("TypeError: Precision Metric: unsupported argument types")
        return 0.0


class F1(Metric):
    name = "f1"

    def __call__(self, predicted: List[str], target: List[str] | str):
        try: 
            if isinstance(predicted, list) and isinstance(target, str):
                target = json.loads(target)
            normalize = lambda s: s.strip().lower()
            pred_set = set(map(normalize, predicted))
            target_set = set(map(normalize, target))
            if not pred_set and not target_set:
                return 1.0  # both empty — define F1 as perfect match
            if not pred_set or not target_set:
                return 0.0

            true_positives = pred_set & target_set
            precision = len(true_positives) / len(pred_set) if pred_set else 0.0
            recall = len(true_positives) / len(target_set) if target_set else 0.0
            if precision + recall == 0:
                return 0.0
            f1 = 2 * precision * recall / (precision + recall)
            return f1
        except Exception as e:
            logging.error(f"F1 Metric: {e}")
            return 0.0

class MeanSquaredError(Metric):
    # This method computes the squared error. The evaluation script is responsible for aggregating.
    name = "mean_squared_error"

    def __call__(self, predicted: str | int | float, target: str | int | float):
        try:
            if isinstance(predicted, str):
                predicted = str_to_float(predicted)
            if isinstance(target, str):
                target = str_to_float(target)
            return (predicted - target) * (predicted - target)
        except Exception as e:
            logging.error(f"MeanSquared Error Metric: {e}")
            return None
    
class MeanAbsoluteError(Metric):
    # This method computes the squared error. The evaluation script is responsible for aggregating.
    name = "mean_absolute_error"

    def __call__(self, predicted: str | int | float, target: str | int | float):
        try:
            if isinstance(predicted, str):
                predicted = str_to_float(predicted)
            if isinstance(target, str):
                target = str_to_float(target)
            return abs(predicted - target)
        except Exception as e:
            logging.error(f"MeanAbsoluteError Metric: {e}")
            return None

class MeanRelativeAbsoluteError(Metric):
    # This method computes the squared error. The evaluation script is responsible for aggregating.
    name = "mean_relative_absolute_error"

    def __call__(self, predicted: str | int | float, target: str | int | float):
        try:
            if isinstance(predicted, str):
                predicted = str_to_float(predicted)
            if isinstance(target, str):
                target = str_to_float(target)
            return abs(predicted - target) / target
        except Exception as e:
            logging.error(f"MeanRelativeAbsoluteError Metric: {e}")
            return 1.0


class BleuScore(Metric):
    name = "bleu"

    def __call__(self, predicted: str, target: str):
        BLEUscore = nltk.translate.bleu_score.sentence_bleu([target], predicted)
        return BLEUscore


class RougeScore(Metric):
    name = "rouge"

    def __call__(self, predicted: str, target: str):
        # Using Rouge-1, the overlap of words
        rouge = rouge_scorer.RougeScorer(['rouge1'])
        results = rouge.score(target=target, prediction=predicted)
        f1 = results['rouge1'].fmeasure
        return f1

class LLMParaphrase(Metric):
    name = "llm_paraphrase"

    def __call__(self, predicted: str, target: str):
        """
        REQUIRES: llm_interface is already initialized.
        """
        llm_interface = GPTInterface()
        is_paraphrase = llm_interface.evaluate_paraphrase(predicted, target)
        if is_paraphrase is not None:
            return int(is_paraphrase)
        return None


class Success(Metric):
    name = "success"

    def __call__(self, predicted: str | int | float, target: str | int | float):
        return int(predicted == target)

def metric_factory(metric_name: str):
    metrics = {
        "precision": Precision,
        "recall": Recall,
        "f1": F1,
        "bleu": BleuScore,
        "rouge": RougeScore,
        "llm_paraphrase": LLMParaphrase,
        "success": Success,
        "mean_squared_error": MeanSquaredError,
        "mean_absolute_error": MeanAbsoluteError,
        "mean_relative_absolute_error": MeanRelativeAbsoluteError,
    }
    if metric_name not in metrics:
        raise ValueError(f"Metric '{metric_name}' not found.")
    return metrics[metric_name]()
