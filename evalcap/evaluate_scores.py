import json
from .coco_caption.pycocotools.coco import COCO
from .coco_caption.pycocoevalcap.eval import COCOEvalCap

class EvaluateCaption():
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.caption_results = {}


    def compute_epoch_score(self, pred_file_path, output_file_path):
        print('\nBeginning Epoch-level evaluation...\n')

        coco = COCO(pred_file_path)
        cocoRes = coco.loadRes(output_file_path)
        cocoEval = COCOEvalCap(coco, cocoRes)

        cocoEval.params['image_id'] = cocoRes.getImgIds()

        cocoEval.evaluate()
        epoch_results = cocoEval.eval

        print('\nEpoch-level Evaluation complete!\n')

        # Only compute caption-level metrics for validation set
        split = split = output_file_path.split('/')[1]
        split = split.split('-')[0]
        
        if split == 'validation':
            print('Beginning Caption-level evaluation... \n')

            img_ids = cocoRes.getImgIds()

            self.caption_results = {}
            for img_id in img_ids:
                cocoEval = COCOEvalCap(coco, cocoRes)
                cocoEval.params['image_id'] = [img_id]
                cocoEval.evaluate()

                self.caption_results[img_id] = cocoEval.evalImgs[0]
        
            print('\nCaption-level Evaluation complete!')

            self.append_caption_scores_to_predictions(output_file_path)

        
        return epoch_results
    

    def append_caption_scores_to_predictions(self, pred_file_path):
        # Load the predictions
        with open(pred_file_path, 'r') as f:
            predictions = json.load(f)

        # Append scores to predictions
        for prediction in predictions:
            img_id = prediction['image_id']
            if img_id in self.caption_results:
                prediction['results'] = self.caption_results[img_id]

        # Save the updated predictions
        output_file_path = pred_file_path.replace('.json', '-caption-metrics.json')
        with open(output_file_path, 'w') as f:
            json.dump(predictions, f, indent=4)

        print(f'\nUpdated predictions with individual scores saved to {output_file_path}')

    
        