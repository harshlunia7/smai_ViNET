import sys
import os
import numpy as np
import cv2
import torch
from models.vinet import ViNet
from scipy.ndimage.filters import gaussian_filter
# from loss import kldiv, cc, nss
import argparse

from torch.utils.data import DataLoader
# from dataloader import DHF1KDataset
# from utils import *
import time
from tqdm import tqdm
from torchvision import transforms, utils
from os.path import join
from PIL import Image

device = torch.device('cpu')
print(device)
def generate_results(args):
    data_path = args.data_path
    ckp_path = args.ckp_path

    clip_size = args.clip_size


    model = ViNet(
        use_upsample=bool(args.decoder_upsample),
        num_hier=args.num_hier,
        num_clips=args.clip_size,
        batch_size=args.batch_size,
        learning_rate=args.lr, 
    )

    model = model.load_from_checkpoint(
        checkpoint_path=args.ckp_path,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    model = model.to(device)
    torch.backends.cudnn.benchmark = False
    model.eval()

    test_video_data = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    test_video_data.sort()

    # if args.start_idx!=-1:
    #     _len = (1.0/float(args.num_parts))*len(list_indata)
    #     list_indata = list_indata[int((args.start_idx-1)*_len): int(args.start_idx*_len)]

    for video_name in test_video_data[1:2]:
        print ('processing ' + video_name, flush=True)
        video_frames = [f for f in os.listdir(os.path.join(data_path, video_name, 'images')) if os.path.isfile(os.path.join(data_path, video_name, 'images', f))]
        video_frames.sort()
        os.makedirs(join(args.experiments_output_dir, video_name), exist_ok=True)

        # process in a sliding window fashion
        if len(video_frames) >= 2*clip_size-1:

            snippet = []
            for i in range(len(video_frames)):
                torch_img, img_size = torch_transform(os.path.join(data_path, video_name, 'images', video_frames[i]))

                snippet.append(torch_img)
                
                if i >= clip_size-1:
                    clip = torch.FloatTensor(torch.stack(snippet, dim=0)).unsqueeze(0)
                    clip = clip.permute((0,2,1,3,4))

                    process(model, clip, data_path, video_name, video_frames[i], args, img_size)
                    # process first (clip_size-1) frames
                    if i < 2*clip_size-2:
                        process(model, torch.flip(clip, [2]), data_path, video_name, video_frames[i-clip_size+1], args, img_size)

                    del snippet[0]
        else:
            print (' more frames are needed')

def torch_transform(path):
    img_transform = transforms.Compose([
            transforms.Resize((224, 384)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
    ])
    img = Image.open(path).convert('RGB')
    sz = img.size
    img = img_transform(img)
    return img, sz

def blur(img):
    k_size = 11
    bl = cv2.GaussianBlur(img,(k_size,k_size),0)
    return torch.FloatTensor(bl)

def process(model, clip, data_path, video_name, frame_no, args, img_size):
    with torch.no_grad():
        smap = model(clip.to(device)).cpu().data[0]
    
    smap = smap.numpy()
    smap = cv2.resize(smap, (img_size[0], img_size[1]))
    smap = blur(smap)
    print("Data type smap", type(smap))
    utils.save_image(smap, join(args.experiments_output_dir, video_name, frame_no), normalize=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckp_path', type=str)
    # parser.add_argument('--nhead',default=4, type=int)
    # parser.add_argument('--num_encoder_layers',default=3, type=int)
    # parser.add_argument('--transformer_in_channel',default=32, type=int)
    parser.add_argument('--experiments_output_dir',default='/ssd_scratch/cvit/rafaelgetto/smap_output', type=str)
    # parser.add_argument('--start_idx',default=-1, type=int)
    # parser.add_argument('--num_parts',default=4, type=int)
    parser.add_argument('--data_path',default='/ssd_scratch/cvit/rafaelgetto/DHF1K/val', type=str)
    parser.add_argument('--decoder_upsample',default=1, type=int)
    parser.add_argument('--num_hier',default=3, type=int)
    parser.add_argument('--lr',default=1e-4, type=int)
    parser.add_argument('--batch_size',default=2, type=int)
    parser.add_argument('--clip_size',default=32, type=int)
    
    args = parser.parse_args()
    print(args)
    generate_results(args)

