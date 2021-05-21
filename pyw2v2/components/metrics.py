from datasets import list_metrics, load_metric


class Metrics:
    
    def __init__(self, *args):
        _all_metrics = list_metrics()
        self._metrics = {m: load_metric(m) for m in args if m in _all_metrics}
        
    def compute(self, predictions, references):
        return {n: m.compute(predictions=predictions, references=references) for n, m in self._metrics.items()}
    
    def list(self):
        return list(self._metrics.keys())