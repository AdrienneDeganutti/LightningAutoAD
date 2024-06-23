import os
import json
import torch
import torch.distributed as dist
from torch.optim import AdamW
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only

class MyLightningModule(pl.LightningModule):
    def __init__(self, args, transformer_model, gpt_model, tokenizer, scorer):
        super().__init__()
        self.args = args
        self.transformer = transformer_model
        self.gpt = gpt_model
        self.tokenizer = tokenizer
        self.scorer = scorer
        self.validation_step_outputs = []
        self.train_step_outputs = []

    def forward(self, caption, visual_frame):
        tokenized_caption = self.tokenizer.tokenize_caption(self.args, caption).to(self.device)
        prefix_vector = self.transformer(visual_frame)
        gpt_output = self.gpt(inputs_embeds=prefix_vector, labels=tokenized_caption)
        loss, logits = gpt_output[:2]
        return loss, logits

    def training_step(self, batch, batch_idx):
        img_ID, caption, visual_frame = batch
        loss, logits = self.forward(caption, visual_frame)

        # Decode prediction:
        token_predictions = torch.argmax(logits, dim=-1)
        train_predictions = [self.tokenizer.tokenizer.decode(g, skip_special_tokens=True) for g in token_predictions]

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        for i in range(len(img_ID)):
            self.train_step_outputs.append({
                'img_ID': img_ID[i],
                'prediction': train_predictions[i]
            })

        return loss

    def validation_step(self, batch, batch_idx):
        img_ID, caption, visual_frame = batch
        loss, logits = self.forward(caption, visual_frame)

        # Decode prediction:
        token_predictions = torch.argmax(logits, dim=-1)
        val_predictions = [self.tokenizer.tokenizer.decode(g, skip_special_tokens=True) for g in token_predictions]

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        for i in range(len(img_ID)):
            self.validation_step_outputs.append({
                'img_ID': img_ID[i],
                'prediction': val_predictions[i]
            })

        return self.validation_step_outputs

    def on_train_epoch_end(self):
        epoch = self.current_epoch

        # Gather predictions from all processes
        all_outputs = self.gather_predictions(self.train_step_outputs)
        
        if torch.distributed.get_rank == 0:
            self.save_predictions(all_outputs, f'output/train-predictions/epoch-{epoch}', epoch, 'train')
        
        torch.distributed.barrier()

        # Compute epoch results
        output_file_path = f'output/train-predictions/epoch-{epoch}/epoch-{epoch}-predictions.json'
        epoch_results = self.scorer.compute_epoch_score(self.args.train_gt_labels, output_file_path, epoch)
        self.log('train_accuracy', epoch_results['CIDEr'], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # Clear outputs
        self.train_step_outputs = []

    def on_validation_epoch_end(self):
        epoch = self.current_epoch

        # Gather predictions from all processes
        all_outputs = self.gather_predictions(self.validation_step_outputs)
        
        if torch.distributed.get_rank == 0:
            self.save_predictions(all_outputs, f'output/validation-predictions/epoch-{epoch}', epoch, 'validation')
        
        torch.distributed.barrier()

        # Compute epoch results
        output_file_path = f'output/validation-predictions/epoch-{epoch}/epoch-{epoch}-predictions.json'
        epoch_results = self.scorer.compute_epoch_score(self.args.val_gt_labels, output_file_path, epoch)
        self.log('val_accuracy', epoch_results['CIDEr'], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # Clear outputs
        self.validation_step_outputs = []


    def save_predictions(self, outputs, output_dir, epoch, split):
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = f'{output_dir}/epoch-{epoch}-predictions.json'

        # Prepare the predictions for JSON saving
        json_predictions = []
        for prediction in outputs:
            amended_prediction = prediction['prediction'].replace('\n', '\\n')  # Escape newline characters
            json_predictions.append({
                'image_id': prediction['img_ID'],
                'caption': amended_prediction
            })

        # Write the predictions to a JSON file
        with open(output_file_path, 'w') as f:  # Use 'w' mode for atomic writes
            json.dump(json_predictions, f, indent=4)

        return output_file_path
    
    def gather_predictions(self, outputs):
        if not dist.is_available() or not dist.is_initialized():
            return outputs

        world_size = dist.get_world_size()
        gathered_outputs = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_outputs, outputs)

        # Flatten the list of lists
        all_outputs = [item for sublist in gathered_outputs for item in sublist]
        return all_outputs

    def configure_optimizers(self):
        combined_parameters = list(self.transformer.parameters()) + list(self.gpt.parameters())
        optimizer = AdamW(combined_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        return optimizer
