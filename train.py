import pytorch_lightning as pl
from pytorch_lightning import Trainer

from dataset import MyDataModule
from lightning_model import LightningModule
from modeling.vision_transformer.model_ad import VideoCaptionModel
from modeling.gpt_model import GPT2LMHeadModel
from configs.load_config import get_custom_args


def main(args):

    transformer_model = VideoCaptionModel(num_latents=args.max_seq_length)
    gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")

    data_module = MyDataModule(args)
    model = LightningModule(args, transformer_model, gpt_model)

    trainer = Trainer(
        max_epochs=args.max_epochs
    )

    trainer.fit(model, data_module)


if __name__ == '__main__':
    args = get_custom_args()
    main(args)
