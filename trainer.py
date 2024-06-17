import os
import json
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
        self.val_predictions = []
        
    
    def forward(self, caption, visual_frame):
        tokenized_caption = self.tokenizer.tokenize_caption(self.args, caption).to(self.device)
        prefix_vector = self.transformer(visual_frame)
        gpt_output = self.gpt(inputs_embeds=prefix_vector, labels=tokenized_caption)
        loss, logits = gpt_output[:2]
        return loss, logits
    
    def training_step(self, batch, batch_idx):
        img_ID, caption, visual_frame = batch

        loss, logits = self.forward(caption, visual_frame)

        batch_acc, predictions = self.scorer.compute_score(img_ID, logits, caption)
        batch_acc_tensor = torch.tensor(batch_acc, device=self.device, dtype=torch.float32)  # Convert to tensor

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_cider', batch_acc_tensor, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        img_ID, caption, visual_frame = batch
        loss, logits = self.forward(caption, visual_frame)

        batch_acc, predictions = self.scorer.compute_score(img_ID, logits, caption)
        batch_acc_tensor = torch.tensor(batch_acc, device=self.device, dtype=torch.float32)  # Convert to tensor

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_cider', batch_acc_tensor, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        for i in range (len(img_ID)):
            self.val_predictions.append({
                'img_ID': img_ID[i],
                'prediction': predictions[img_ID[i]]})
        
        return loss
    
    def on_validation_epoch_end(self):
        epoch = self.current_epoch

        with open(f'output/val_results_epoch_{epoch}.json', 'w') as f:
            json.dump(self.val_predictions, f, indent=4)
        self.val_predictions = []   #Clear for the next epoch

    
    def configure_optimizers(self):
        #Combine parameters and initialize optimizer
        combined_parameters = list(self.transformer.parameters()) + list(self.gpt.parameters())
        return AdamW(combined_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)