import numpy as np
from rouge import Rouge 

class Evaluator(object):
    def __init__(self):
        print('Evaluator initialning!')
        self.rouge_ = Rouge()
        
    def rouge(self, refs, hyps):
        scores = self.rouge_.get_scores(hyps, refs, avg=True)
        return [scores['rouge-1']['f'], scores['rouge-2']['f'], scores['rouge-l']['f']]

