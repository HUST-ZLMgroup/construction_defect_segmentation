This repository contains part of the code for the paper **In-domain Adapted SAM with YOLO-guided Detection Prompting for Zero-shot Construction Defect Segmentation.**
## 1. Dataset preparation

### 1. Industrial defect dataset
The industrial defect dataset is used for fine-tuning SAM.

| Dataset | Classes | Major Class | Images | Masks | Download |
|---|---:|---|---:|---:|---|
| Magnetic Tile | 1 | Magnetic tile | 392 | 392 | [GitHub](https://github.com/abin24/Saliency-detection-toolbox) |
| MSD-Seg2 | 6 | Metal | 561 | 561 | [GitHub](https://github.com/bbbbby-99/TGRNet-Surface-Defect-Segmentation) |
| MVTec AD | 15 | Industrial | 1,258 | 1,258 | [Official Website](https://www.mvtec.com/research-teaching/datasets/mvtec-ad/downloads) |
| Rail | 1 | Rail | 1,107 | 1,107 | [GitHub](https://github.com/FreddyDylan/Dataset) |
| SD-Saliency-900 | 1 | Steel | 900 | 900 | [Kaggle](https://www.kaggle.com/datasets/alex000kim/sdsaliency900?resource=download) |
| ZJU-Leaper | 1 | Fabric | 23,706 | 23,706 | [Official Website](http://www.qaas.zju.edu.cn/zju-leaper/) |
| Structural Defects | 1 | Structure | 104 | 104 | [Official Website](http://dmi.uib.es/xbonnin/resources) |
| **Total** | — | — | **28,028** | **28,028** | — |

### 2. Construction defect dataset
The construction defect dataset is used to train YOLO for generating bounding box prompts and to evaluate the complete YOLO+SAM pipeline.

| Defect Type | Images | Annotations | Download |
|---|---:|---:|---|
| Crack | 500 | 500 | [Roboflow](https://universe.roboflow.com/construction-defect-dataset/crack-uacac-l1okn) |
| Damp Stain | 985 | 985 | [Roboflow](https://universe.roboflow.com/construction-defect-dataset/damp-stain-cvkc3) |
| Efflorescence | 619 | 619 | [Roboflow](https://universe.roboflow.com/construction-defect-dataset/efflorescence-hsa7w-q5tfn) |
| Leakage Water Stain | 500 | 500 | [Roboflow](https://universe.roboflow.com/construction-defect-dataset/leakagewaterstain-r4rg9-wnf52) |
| Rebar Exposure | 701 | 701 | [Roboflow](https://universe.roboflow.com/construction-defect-dataset/rebar-seg-sb7af) |
| Roof Damage | 638 | 638 | [Roboflow](https://universe.roboflow.com/construction-defect-dataset/roof-damage-ebuyf-vsenw) |
| Spalling | 719 | 719 | [Roboflow](https://universe.roboflow.com/construction-defect-dataset/spalling-nkeeh-vz66s) |
| **Total** | **4,662** | **4,662** | — |

## 2. Train

### 1. YOLO
For YOLO, we directly adopt the official YOLOv5 implementation without any modifications, as it is used solely to generate bounding box prompts for SAM. The model is trained using the default training command provided by the official YOLOv5 repository. 

Therefore, only an illustrative example is provided here.
```
python train.py --img 640 --batch 16 --epochs 100  --data construction_defect.yaml  --weights yolov5x.pt
```
### 2. SAM
This involves fine-tuning SAM on a custom industrial defect dataset.

Simply follow the official SAM installation instructions to set up the environment.
```
python -m torch.distributed.launch --nproc_per_node=<num_gpus> train.py --checkpoint <path/to/checkpoint> --model-type <model_type> --output <path/to/output>
```
Multi-GPU training is recommended for improved training efficiency. In this study, two NVIDIA RTX 3090 GPUs (24 GB each) are used. 

With a batch size of 4 per GPU, ViT-B requires approximately 11 GB of GPU memory and takes about 30 minutes per epoch. ViT-L requires approximately 16 GB of GPU memory and takes around 1 hour per epoch, while ViT-H requires approximately 18.9 GB of GPU memory and takes approximately 80 minutes per epoch under the same batch size setting. 

It should be noted that training time depends on the dataset size, and the values reported here are provided for reference only.
## 3. Evaluation
This is the evaluation of the complete YOLO+SAM framework.
```
bash test_ap_on_yolov5_sam.sh
```
These are the evaluation results on the construction defect dataset.

| Model | IoU | Dice |
|:---|:---:|:---:|
| SAM | 0.4127 | 0.5843 |
| SAM-Adapter | 0.3412 | 0.5088 |
| MedSAM | 0.5196 | 0.6839 |
| **Ours** | **0.8247** | **0.9040** |
