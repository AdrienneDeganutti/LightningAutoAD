import pytorch_lightning as pl
from pytorch_lightning import Trainer

from dataset import MyDataModule
from trainer import LightningModule
from modeling.tokenizer.caption_tokenizer import TokenizerHandler
from modeling.vision_transformer.model_ad import VideoCaptionModel
from evalcap.evaluate_scores import EvaluateCaption
from modeling.gpt_model import GPT2LMHeadModel
from configs.load_config import get_custom_args


def main(args):

    tokenizer = TokenizerHandler()
    transformer_model = VideoCaptionModel(num_latents=args.max_seq_length)
    gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
    scorer = EvaluateCaption(args, tokenizer)

    data_module = MyDataModule(args)
    
    model = LightningModule(args, transformer_model, gpt_model, tokenizer, scorer)



    trainer = Trainer(
        accelerator='gpu', 
        devices=1,
        max_epochs=args.max_epochs
    )

    trainer.fit(model, data_module)


if __name__ == '__main__':
    args = get_custom_args()
    main(args)
