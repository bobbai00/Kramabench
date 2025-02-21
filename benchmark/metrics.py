
"""
TDOO: Instantiate a Metric class based on name.
"""

class Metric:
    name = "Metric"

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, predicted:dict, target:dict):
        raise NotImplementedError("Metric must implement __call__ method!")


class Accuracy(Metric):

    name = "Accuracy"

    def __call__(self, predicted:dict, target:dict):
        return 0.0
        # return accuracy_score(predicted["answer"], target["answer"])

class Precision(Metric):

    name = "Precision"

    def __call__(self, predicted:dict, target:dict):
        # Figure out how to compute precision of individual queries
        return 0
        # return precision_score(predicted["answer"], target["answer"])
    


class Recall(Metric):

    name = "Recall"

    def __call__(self, predicted:dict, target:dict):
        return 0.0
        # return recall_score(predicted["answer"], target["answer"])


class F1(Metric):

    name = "F1"

    def __call__(self, predicted:dict, target:dict):
        return 0.0
        # return f1_score(predicted["answer"], target["answer"])


class BleuScore(Metric):

    name = "BLEU"

    def __call__(self, predicted:dict, target:dict):
        return 0.0
        # return bleu_score(predicted["answer"], target["answer"])


class RougeScore(Metric):

    name = "ROUGE"

    def __call__(self, predicted:dict, target:dict):
        return 0.0
        # return rouge_score(predicted["answer"], target["answer"])


class Success(Metric):

    name = "Success"

    def __call__(self, predicted:dict, target:dict):
        return 0.0
        # return success_score(predicted["answer"], target["answer"])
