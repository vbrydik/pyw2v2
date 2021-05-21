from datasets import list_metrics, load_metric
from numpy.lib.arraysetops import isin


class Metrics:
    
    def __init__(self, *args):
        _all_metrics = list_metrics()
        self._metrics = {m: load_metric(m) for m in args if m in _all_metrics and isinstance(m, str)}
        
    def compute(self, predictions, references) -> dict:
        return {n: m.compute(predictions=predictions, references=references) for n, m in self._metrics.items()}
    
    def list(self) -> list:
        return list(self._metrics.keys())