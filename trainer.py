import os
import csv
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl

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

        for i in range (len(img_ID)):
            self.train_step_outputs.append({
                'img_ID': img_ID[i],
                'prediction': train_predictions[i]})

        return loss
    

    def validation_step(self, batch, batch_idx):
        img_ID, caption, visual_frame = batch
        loss, logits = self.forward(caption, visual_frame)

        # Decode prediction:
        token_predictions = torch.argmax(logits, dim=-1)
        val_predictions = [self.tokenizer.tokenizer.decode(g, skip_special_tokens=True) for g in token_predictions]

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        for i in range (len(img_ID)):
            self.validation_step_outputs.append({
                'img_ID': img_ID[i],
                'prediction': val_predictions[i]})
        
        return self.validation_step_outputs
    

    def on_train_epoch_end(self):
        epoch = self.current_epoch

        os.makedirs('output/train-predictions', exist_ok=True)
        output_file_path = f'output/train-predictions/epoch-{epoch}.tsv'

        # Write the val_predictions to a TSV file
        with open(output_file_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['img_ID', 'prediction'])

            for prediction in self.train_step_outputs:
                amended_prediction = prediction['prediction'].replace('\n', '\\n')
                writer.writerow([prediction['img_ID'], amended_prediction])
    
        self.train_step_outputs = []  # Clear for the next epoch
        return super().on_train_epoch_end()
    

    def on_validation_epoch_end(self):
        epoch = self.current_epoch

        os.makedirs('output/validation-predictions', exist_ok=True)
        output_file_path = f'output/validation-predictions/epoch-{epoch}.tsv'

        # Write the val_predictions to a TSV file
        with open(output_file_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['img_ID', 'prediction'])

            for prediction in self.validation_step_outputs:
                amended_prediction = prediction['prediction'].replace('\n', '\\n')
                writer.writerow([prediction['img_ID'], amended_prediction])
    
        self.validation_step_outputs = []  # Clear for the next epoch

    
    def configure_optimizers(self):
        #Combine parameters and initialize optimizer
        combined_parameters = list(self.transformer.parameters()) + list(self.gpt.parameters())
        return AdamW(combined_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)