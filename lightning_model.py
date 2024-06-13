import torch
import torch.nn as nn
from torch.optim import AdamW
import pytorch_lightning as pl

class LightningModule(pl.LightningModule):
    def __init__(self, args, transformer_model, gpt_model):
        super().__init__()
        self.args = args
        self.transformer = transformer_model
        self.gpt = gpt_model
        
        #Combine parameters and initialize optimizer
        combined_parameters = list(self.transformer.parameters()) + list(self.gpt.parameters())
        self.optimizer = AdamW(combined_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        
    
    def forward(self, x):
        prefix_vector = self.transformer(visual_frame)
        gpt_output = self.gpt(input_embeds=prefix_vector, labels=tokenized_caption)
        loss, logits = gpt_output[:2]
        return loss, logits
    
    def training_step(self, batch, batch_idx):
        img_ID, caption, visual_frame = batch
        
        tokenized_caption = tokenizer.tokenize_caption(self.args, caption)

        prefix_vector = self.transformer(visual_frame)
        gpt_output = self.gpt(input_embeds=prefix_vector, labels=tokenized_caption)
        loss, logits = gpt_output[:2]

        batch_score = compute_score_with_logits(logits, tokenized_caption)
        batch_acc = torch.mean(batch_score.float())

        loss_dict = {'loss': float(loss), 'acc': float(batch_acc)}

        self.log_dict(loss_dict)

        return loss
    
    def validation_step(self, batch, batch_idx):
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)