import cv2
import os
import glob
from PIL import Image
import numpy as np

data_dir = "/ssd_scratch/cvit/rafaelgetto/DIEM/"
annotation_data_dir = "/ssd_scratch/cvit/rafaelgetto/DIEM/annotations"
# file_obj = open("zero_maps.txt", "w")
video_names = os.listdir(annotation_data_dir)
for v in video_names:
    audio_file = glob.glob(os.path.join(data_dir, "video_audio", v, "./*wav"))
    try:
        assert len(audio_file) == 1
    except:
        print(v)
