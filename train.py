# Basic packages
import argparse
import os
import torch

# 3rd part utility packages
from tqdm import tqdm

# pytorch library and modules

# from torchsummary import summary

# torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

from models.vinet import ViNet
from models.avinet import AViNet
from data.module import SaliencyDataModule

parser = argparse.ArgumentParser()

# General Model Train Hyperparameters
parser.add_argument("--epochs", default=40, type=int)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--log_interval", default=5, type=int)
parser.add_argument("--no_workers", default=4, type=int)

parser.add_argument("--l1", default=False, type=bool)
parser.add_argument("--l1_coeff", default=1.0, type=float)
parser.add_argument("--lr_sched", default=False, type=bool)
parser.add_argument("--optim", default="Adam", type=str)

parser.add_argument("--kldiv", default=True, type=bool)
parser.add_argument("--kldiv_coeff", default=1.0, type=float)

# Dataset Realted
parser.add_argument("--dataset", default="DHF1K", type=str)
parser.add_argument("--experiment_name", default="ViNet_epoch_100", type=str)
parser.add_argument(
    "--data_directory", default="/ssd_scratch/cvit/rafaelgetto", type=str
)

# ViNet Model specific hyperparameters
parser.add_argument("--clip_size", default=32, type=int)
parser.add_argument("--decoder_upsample", default=1, type=int)
parser.add_argument("--frame_no", default="last", type=str)
parser.add_argument("--load_weight", default="None", type=str)
parser.add_argument("--fusing_method", default="concat", type=str)
parser.add_argument("--num_hier", default=3, type=int)
parser.add_argument("--load_encoder_weights", default=False, type=bool)
parser.add_argument("--alternate", default=1, type=int)
parser.add_argument("--split", default=-1, type=int)
parser.add_argument("--use_sound", default=False, type=bool)
parser.add_argument("--use_transformer", default=False, type=bool)

args = parser.parse_args()
print(args)

logger = TensorBoardLogger("ViNet_Logs", name=f"ViNet_Logs_{args.experiment_name}")
dm = SaliencyDataModule(
    dataset_name=args.dataset,
    root_data_dir=args.data_directory,
    clip_length=args.clip_size,
    batch_size=args.batch_size,
    num_workers=args.no_workers,
    use_sound=args.use_sound,
)
if args.use_sound == False:
    model = ViNet(
        use_upsample=bool(args.decoder_upsample),
        num_hier=args.num_hier,
        num_clips=args.clip_size,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
else:
    model = AViNet(
        use_upsample=bool(args.decoder_upsample),
        num_hier=args.num_hier,
        num_clips=args.clip_size,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        fusing_method=args.fusing_method,
        use_transformer=bool(args.use_transformer),
    )


# print(summary(model, (3, 48, 224, 384)))
S3D_weight_file = "./S3D_kinetics400.pt"

if args.load_encoder_weights and args.use_sound == False:
    if os.path.isfile(S3D_weight_file):
        print("loading weight file")
        weight_dict = torch.load(
            S3D_weight_file
        )  # ,map_location=torch.device('cpu'))#,map_location=torch.device('cpu'))
        model_dict = model.backbone_encoder.state_dict()
        for key, value in weight_dict.items():
            if "module" in key:
                key = ".".join(key.split(".")[1:])

            if "base." in key:
                base_num = int(key.split(".")[1])
                sn_list = [0, 5, 8, 14]
                sn = sn_list[0]
                if base_num >= sn_list[1] and base_num < sn_list[2]:
                    sn = sn_list[1]
                elif base_num >= sn_list[2] and base_num < sn_list[3]:
                    sn = sn_list[2]
                elif base_num >= sn_list[3]:
                    sn = sn_list[3]
                key = ".".join(key.split(".")[2:])
                key = "base%d.%d." % (sn_list.index(sn) + 1, base_num - sn) + key

            if key in model_dict:
                if value.size() == model_dict[key].size():
                    model_dict[key].copy_(value)
                else:
                    print(" size? " + key, value.size(), model_dict[key].size())
            else:
                print(" name? " + key)
        print(" loaded")
        model.backbone_encoder.load_state_dict(model_dict)
    else:
        print("weight file?")
elif args.use_sound == True and args.load_weight != "None":
    model.visual_model.load_from_checkpoint(
        checkpoint_path=args.load_weight,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
elif args.use_sound == False and args.load_weight != "None":
    model.load_from_checkpoint(
        checkpoint_path=args.load_weight,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
trainer = pl.Trainer(
    accelerator="gpu", devices="auto", logger=logger, strategy="ddp", max_epochs=105
)
trainer.fit(model, dm)
