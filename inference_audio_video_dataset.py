import os
import numpy as np
import cv2
import torch
from models.vinet import ViNet
from models.avinet import AViNet
import argparse

from torchvision import transforms, utils
# import torchaudio
from os.path import join
from PIL import Image

device = torch.device('cpu')
print(device)
def generate_results(args):
    data_path = args.data_path
    clip_size = args.clip_size
    video_to_generate = args.video_to_generate_for

    if args.trained_on_sound == False and args.trained_without_sound == True and args.trained_on_noise == False:
        model = ViNet(
            use_upsample=bool(args.decoder_upsample),
            num_hier=args.num_hier,
            num_clips=args.clip_size,
            batch_size=args.batch_size,
            learning_rate=args.lr, 
        )
    elif (args.trained_on_sound == True or args.trained_on_noise == True) and args.trained_without_sound == False:
        model = AViNet(
            use_upsample=bool(args.decoder_upsample),
            num_hier=args.num_hier,
            num_clips=args.clip_size,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            fusing_method=args.fusion_method,
            use_transformer=bool(args.use_transformer),
        )
    
    model = model.load_from_checkpoint(
        checkpoint_path=args.ckp_path,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    model = model.to(device)
    torch.backends.cudnn.benchmark = False
    model.eval()

    test_video_names = [d for d in os.listdir(os.path.join(data_path, "annotations")) if os.path.isdir(os.path.join(data_path, "annotations", d))]
    if video_to_generate not in test_video_names:
        print("NOTE:: Video not present in data!")
        print(test_video_names)
        return
    
    print ('processing ' + video_to_generate, flush=True)
    video_frames = [f for f in os.listdir(os.path.join(data_path, "video_frames", video_to_generate)) if os.path.isfile(os.path.join(data_path, "video_frames", video_to_generate, f))]
    video_frames.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    os.makedirs(join(args.experiments_output_dir, video_to_generate), exist_ok=True)

    # if args.trained_on_sound == True:
    #     audio_data = 
    # process in a sliding window fashion
    if len(video_frames) >= 2*clip_size-1:

        snippet = []
        for i in range(len(video_frames)):
            print(f"Processing frame: {i}/{len(video_frames)}")
            torch_img, img_size = torch_transform(os.path.join(data_path, "video_frames", video_to_generate, video_frames[i]))

            snippet.append(torch_img)
            
            if i >= clip_size-1:
                clip = torch.FloatTensor(torch.stack(snippet, dim=0)).unsqueeze(0)
                clip = clip.permute((0,2,1,3,4))

                process(model, clip, video_to_generate, video_frames[i], args, img_size)
                # process first (clip_size-1) frames
                if i < 2*clip_size-2:
                    process(model, torch.flip(clip, [2]), video_to_generate, video_frames[i-clip_size+1], args, img_size)

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

def process(model, clip, video_name, frame_no, args, img_size):
    with torch.no_grad():
        smap = model(clip.to(device)).cpu().data[0]
    
    smap = smap.numpy()
    smap = cv2.resize(smap, (img_size[0], img_size[1]))
    smap = blur(smap)
    utils.save_image(smap, join(args.experiments_output_dir, video_name, frame_no), normalize=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckp_path', type=str)
    parser.add_argument('--experiments_output_dir',default='/ssd_scratch/cvit/rafaelgetto/smap_output', type=str)
    parser.add_argument('--data_path',default='/ssd_scratch/cvit/rafaelgetto/AVAD/', type=str)
    parser.add_argument('--video_to_generate_for',default='V17_Soccer1', type=str)
    parser.add_argument('--decoder_upsample',default=1, type=int)
    parser.add_argument('--num_hier',default=3, type=int)
    parser.add_argument('--lr',default=1e-4, type=int)
    parser.add_argument("--use_transformer", default=False, type=bool)
    parser.add_argument("--trained_on_sound", default=False, type=bool)
    parser.add_argument("--trained_on_noise", default=False, type=bool)
    parser.add_argument("--trained_without_sound", default=False, type=bool)
    parser.add_argument('--fusion_method',default='concat', type=str)
    parser.add_argument('--batch_size',default=2, type=int)
    parser.add_argument('--clip_size',default=32, type=int)

    args = parser.parse_args()
    print(args)

    generate_results(args)

