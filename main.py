import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger

from dataset import MyDataModule
from trainer import MyLightningModule
from modeling.tokenizer.caption_tokenizer import TokenizerHandler
from modeling.vision_transformer.model_ad import VideoCaptionModel
from evalcap.evaluate_scores import EvaluateCaption
from modeling.gpt_model import GPT2LMHeadModel
from configs.load_config import get_custom_args


def main(args):

    #torch.manual_seed(args.seed)
    #seed_everything(args.seed)

    wandb_logger = WandbLogger(project="Lightning-AutoAD")

    tokenizer = TokenizerHandler()
    transformer_model = VideoCaptionModel(num_latents=args.max_seq_length)
    gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
    scorer = EvaluateCaption(args, tokenizer)

    data_module = MyDataModule(args)
    
    model = MyLightningModule(args, transformer_model, gpt_model, tokenizer, scorer)

    trainer = Trainer(
        accelerator='gpu',
        devices=args.num_devices,
        strategy=DDPStrategy(find_unused_parameters=True),
        use_distributed_sampler = False,        #Stops the train_dataset from shuffling
        max_epochs=args.max_epochs,
        logger=wandb_logger
    )

    trainer.fit(model, data_module)


if __name__ == '__main__':
    args = get_custom_args()
    main(args)
