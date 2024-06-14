import torch
from pycocoevalcap.cider.cider import Cider

class EvaluateCaption():
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.cider_scorer = Cider()

    def compute_score(self, img_ID, logits, GT_caption):

        token_predictions = torch.argmax(logits, dim=-1)
        predictions = [self.tokenizer.tokenizer.decode(g, skip_special_tokens=True) for g in token_predictions]

        # Prepare references and candidates for CIDEr computation
        ground_truth = {img_ID[i]: [GT_caption[i]] for i in range(len(img_ID))}
        candidates = {img_ID[i]: [predictions[i]] for i in range(len(img_ID))}

        # Compute CIDEr score
        cider_score, _ = self.cider_scorer.compute_score(ground_truth, candidates)
    
        return cider_score