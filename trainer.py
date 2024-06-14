import torch
import torch.nn as nn
from torch.optim import AdamW
import pytorch_lightning as pl

class LightningModule(pl.LightningModule):
    def __init__(self, args, transformer_model, gpt_model, tokenizer, scorer):
        super().__init__()
        self.args = args
        self.transformer = transformer_model
        self.gpt = gpt_model
        self.tokenizer = tokenizer
        self.scorer = scorer
        
    
    def forward(self, tokenized_caption, visual_frame):
        prefix_vector = self.transformer(visual_frame)
        gpt_output = self.gpt(inputs_embeds=prefix_vector, labels=tokenized_caption)
        loss, logits = gpt_output[:2]
        return loss, logits
    
    def training_step(self, batch, batch_idx):
        img_ID, caption, visual_frame = batch
        
        tokenized_caption = self.tokenizer.tokenize_caption(self.args, caption)

        loss, logits = self.forward(tokenized_caption, visual_frame)

        batch_score = compute_score_with_logits(logits, tokenized_caption)
        batch_acc = torch.mean(batch_score.float())

        loss_dict = {'loss': float(loss), 'acc': float(batch_acc)}

        self.log_dict(loss_dict)

        return loss
    
    def validation_step(self, batch, batch_idx):
        img_ID, caption, visual_frame = batch
        #tokenized_caption = self.tokenizer.tokenize_caption(self.args, caption)

        prefix_vector = self.transformer(visual_frame)
        outputs = self.gpt(inputs_embeds=prefix_vector, )      #No caption given for validation
        logits = outputs[0]

        batch_acc = self.scorer.compute_score(img_ID, logits, caption)

        self.log('validation_cider', batch_acc)
        
        return batch_acc
    
    def configure_optimizers(self):
        #Combine parameters and initialize optimizer
        combined_parameters = list(self.transformer.parameters()) + list(self.gpt.parameters())
        return AdamW(combined_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)