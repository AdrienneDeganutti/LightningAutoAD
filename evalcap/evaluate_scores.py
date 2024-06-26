import json
from .coco_caption.pycocotools.coco import COCO
from .coco_caption.pycocoevalcap.eval import COCOEvalCap

class EvaluateCaption():
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

    def compute_epoch_score(self, gt_file_path, pred_file_path):

        coco = COCO(gt_file_path)
        cocoRes = coco.loadRes(pred_file_path)
        cocoEval = COCOEvalCap(coco, cocoRes)

        cocoEval.params['image_id'] = cocoRes.getImgIds()

        cocoEval.evaluate()
        results = cocoEval.eval

        return results
    
        