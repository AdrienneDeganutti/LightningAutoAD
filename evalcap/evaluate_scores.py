import os
import json
import torch
import torch.distributed as dist
from .coco_caption.pycocotools.coco import COCO
from .coco_caption.pycocoevalcap.eval import COCOEvalCap
from pytorch_lightning.utilities.rank_zero import rank_zero_only

class EvaluateCaption():
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.caption_results = {}

    def compute_epoch_score(self, pred_file_path, output_file_path, epoch):
        
        rank = dist.get_rank()
        print('######')
        print(f'Beginning Epoch-level evaluation for RANK {rank}')
        print('######')

        coco = COCO(pred_file_path)
        cocoRes = coco.loadRes(output_file_path)
        cocoEval = COCOEvalCap(coco, cocoRes)

        cocoEval.params['image_id'] = cocoRes.getImgIds()

        cocoEval.evaluate()
        epoch_results = cocoEval.eval

        # Only compute caption-level metrics for validation set
        split = split = output_file_path.split('/')[1]
        split = split.split('-')[0]

        # Gather results from all processes
        gathered_results = self.gather_results(epoch_results)
        
        # Only the rank zero process writes the results
        if torch.distributed.get_rank == 0:
            self.save_epoch_results(gathered_results, split, epoch)
        
        torch.distributed.barrier()
        
        if split == 'validation':
            
            rank = dist.get_rank()
            print('######')
            print(f'Beginning Caption-level evaluation for RANK {rank}')
            print('######')

            img_ids = cocoRes.getImgIds()

            self.caption_results = {}
            for img_id in img_ids:
                cocoEval = COCOEvalCap(coco, cocoRes)
                cocoEval.params['image_id'] = [img_id]
                cocoEval.evaluate()

                self.caption_results[img_id] = cocoEval.evalImgs[0]
    
            if torch.distributed.get_rank == 0:
                self.append_caption_scores_to_predictions(output_file_path)

        torch.distributed.barrier()
        return epoch_results
    

    def gather_results(self, local_results):
        if not dist.is_available() or not dist.is_initialized():
            return local_results

        world_size = dist.get_world_size()
        gathered_results = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_results, local_results)

        if dist.get_rank() == 0:
            combined_results = {}
            for result in gathered_results:
                for key, value in result.items():
                    if key not in combined_results:
                        combined_results[key] = []
                    combined_results[key].append(value)

            aggregated_results = {key: sum(values) / len(values) for key, values in combined_results.items()}
            return aggregated_results
        return None

    def save_epoch_results(self, epoch_results, split, epoch):
        output_dir = f'output/{split}-predictions/epoch-{epoch}'
        output_results_path = f'{output_dir}/epoch-{epoch}-epoch-metrics.json'
        os.makedirs(output_dir, exist_ok=True)
        with open(output_results_path, 'w') as fp:
            json.dump(epoch_results, fp, indent=4)


    def append_caption_scores_to_predictions(self, pred_file_path):
        with open(pred_file_path, 'r') as f:
            predictions = json.load(f)

        for prediction in predictions:
            img_id = prediction['image_id']
            if img_id in self.caption_results:
                prediction['results'] = self.caption_results[img_id]

        output_file_path = pred_file_path.replace('.json', '-caption-metrics.json')
        with open(output_file_path, 'w') as f:
            json.dump(predictions, f, indent=4)

        print(f'\nUpdated predictions with individual scores saved to {output_file_path}')