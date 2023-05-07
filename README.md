## Code Repo Structure
```
├── README.md
├── S3D_kinetics400.pt
├── ada_train_avinet.sh
├── ada_train_avinet_with_noise.sh
├── ada_train_avinet_without_sound.sh
├── ada_train_vinet.sh
├── add_noise.py - **Preprocessing script,adding noise to audio files**
├── data
│   ├── __init__.py
│   ├── datasets.py - **Dataset class defined for visual and audio-visual data**
│   └── module.py - **Pytorch Lightning Datamodule which feeds data to trainer**
├── evaluation
│   ├── __init__.py
│   └── modules.py - **All Loss functions and evaluation metric**
├── get_all_zero_maps.py - **To check if any saliency map is completely null (0)**
├── get_non_audio_video_names.py - **To check if any video doesn't have audio file**
├── inference_audio_video_dataset.py - **smap prediction for audio video dataset**
├── inference_visual_dataset.py - **smap prediction for video dataset**
├── models
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-36.pyc
│   │   └── base_modules.cpython-36.pyc
│   ├── avinet.py - **AViNet model**
│   ├── base_modules.py - **encoder, decoder, soundnet and fusion via transformer**
│   ├── base_system.py - **Base class of all models defining the optimizer and lr**
│   ├── decoder_config.yaml - **Different architecture configs of Encoder-Decoder**
│   ├── utils.py - **Encoder different submodules**
│   └── vinet.py - **ViNet model**
├── run_predict.py
├── soundnet8_final.pth
├── train.py - Train script
├── train_test.sh
└── zero_maps.txt
```

## Train commands are in respective SBATCH files (Files starting with ada_)

## Inference Command examples
*python inference_audio_video_dataset.py --ckp_path ./ViNet_Logs/ViNet_Logs_ViNet_without_sound_AVAD_earlystopping/version_0/checkpoints/epoch\=64-step\=519-val_Loss\=0.6914-val_cc_loss\=0.7214-val_similarity\=0.5506.ckpt --trained_without_sound True*

*python inference_visual_dataset.py --ckp_path ViNet_Logs/ViNet_Logs_ViNet_early_stopping/version_0/checkpoints/epoch\=64-step\=9749-val_Loss\=1.2245-val_cc_loss\=0.5153-val_similarity\=0.4076.ckpt*

## Checkpoints Drive Link
https://iiitaphyd-my.sharepoint.com/:f:/g/personal/harsh_lunia_students_iiit_ac_in/Em0tmK0aYytPsvpkMQvPip8BJo7CoGAsFM3eU7uBgodRtQ?e=DqaLbO

## Predicted Saliency Maps (LEFT) and Ground Truth (RIGHT) Videos
https://iiitaphyd-my.sharepoint.com/:f:/g/personal/harsh_lunia_students_iiit_ac_in/ElQrHnpzGKJPk39VmzNCmSkBkZRkCnDNYX1Ukkz3L5rfcg?e=FTnJry
