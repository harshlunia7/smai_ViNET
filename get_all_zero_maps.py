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
    print(v)
    ann_maps = glob.glob(os.path.join(annotation_data_dir, v, "maps", "./*.jpg"))
    fix_files = glob.glob(os.path.join(annotation_data_dir, v, "./fixMap*.mat"))
    frame_files = glob.glob(os.path.join(data_dir, "video_frames", v, "./*.jpg"))
    #    print(len(ann_maps), len(fix_files), len(frame_files))
    assert len(ann_maps) == len(frame_files)
    assert len(ann_maps) == len(fix_files)
    ann_maps.sort(key=lambda x: int(x.split("/")[-1].split("_")[-1].split(".")[0]))
    fix_files.sort(key=lambda x: int(x.split("/")[-1].split("_")[-1].split(".")[0]))
    frame_files.sort(key=lambda x: int(x.split("/")[-1].split("_")[-1].split(".")[0]))
    i = 1
    for map_file, fix_file, frame_file in zip(ann_maps, fix_files, frame_files):
        ann_map_index = map_file.split("/")[-1].split("_")[-1].split(".")[0]
        fix_file_index = fix_file.split("/")[-1].split("_")[-1].split(".")[0]
        frame_file_index = frame_file.split("/")[-1].split("_")[-1].split(".")[0]
        assert ann_map_index == fix_file_index
        assert ann_map_index == frame_file_index
        os.system(
            f"mv {fix_file} {os.path.join(data_dir, 'annotations', v, f'fixMap_{str(i).zfill(5)}.mat')}"
        )
        os.system(
            f"mv {map_file} {os.path.join(data_dir, 'annotations', v, 'maps', f'eyeMap_{str(i).zfill(5)}.jpg')}"
        )
        os.system(
            f"mv {frame_file} {os.path.join(data_dir, 'video_frames', v, f'img_{str(i).zfill(5)}.jpg')}"
        )
        i += 1
        # gt = np.array(Image.open(m).convert("L"))
        # gt = gt.astype("float")
        # index = m.split('/')[-1].split('_')[-1].split('.')[0]
        # assert gt.max() != 0

        # try:
        #     assert gt.max() != 0
        # except:
        #     os.system(f"rm {os.path.join(data_dir, 'annotations', v, f'fixMap_{index}.mat')}")
        #     os.system(f"rm {os.path.join(data_dir, 'annotations', v, 'maps', f'eyeMap_{index}.jpg')}")
        #     os.system(f"rm {os.path.join(data_dir, 'video_frames', v, f'img_{index}.jpg')}")

# file_obj.close()
