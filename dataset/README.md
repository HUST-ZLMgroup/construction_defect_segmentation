### 1. industrial defect dataset
To enable zero-shot transfer for construction defect segmentation, we construct a unified industrial defect dataset by integrating seven publicly available defect segmentation datasets. The resulting dataset is used to fine-tune SAM, allowing it to learn generic defect representations that can be transferred from industrial defect domains to construction defect scenarios.

The unified industrial defect dataset consists of seven publicly available datasets. The dataset sources and statistics after preprocessing are summarized below.

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

The datasets exhibit substantial differences in scale and defect distribution, resulting in a highly imbalanced training corpus. Nevertheless, this diversity provides SAM with defect examples from different industrial domains, supporting the learning of transferable defect representations.

To ensure consistent image–mask supervision, only samples with valid pixel-level ground-truth masks are retained. The preprocessing procedures for each dataset are described below.

- **Magnetic Tile:** The `MT_Free` subset is excluded because it does not provide effective defect masks.
- **MSD-Seg2, Rail, and SD-Saliency-900:** These datasets are used in their original formats, with images and corresponding masks reorganized into `images` and `gt` directories.
- **MVTec AD:** Normal samples from `train/good` and `test/good` are excluded. All remaining defective samples across object categories are merged into a unified image–mask dataset. Ground-truth masks are binarized, and category information is discarded during training.
- **Structural Defects:** Only the Plus version is used.
- **ZJU-Leaper:** Only images with corresponding ground-truth masks are retained and reorganized into the unified dataset structure.

After preprocessing, the images and their corresponding ground-truth masks can be organized into the following directory structure:

```text
Unified_Industrial_Defect_Dataset/
├── images/
│   ├── image_00001.png
│   ├── image_00002.png
│   └── ...
└── gt/
    ├── image_00001.png
    ├── image_00002.png
    └── ...
```

Each image is paired with its corresponding pixel-level ground-truth mask.

### 2. construction defect dataset
To evaluate zero-shot transfer to real-world construction defect scenarios, we construct a multi-category construction defect dataset by integrating seven publicly available defect datasets. The resulting dataset covers seven representative construction defect categories, including Crack, Damp Stain, Efflorescence, Leakage Water Stain, Rebar Exposure, Roof Damage, and Spalling.

Unlike the industrial defect dataset used for SAM fine-tuning, the construction defect dataset is used **exclusively for zero-shot evaluation**, enabling the assessment of domain transfer capability from industrial defect knowledge to construction defect segmentation.

The construction defect dataset consists of seven publicly available datasets. The dataset sources and statistics are summarized below.

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

The dataset contains **4,662 images and 4,662 annotations**, with one defect instance annotated per image. Each category contains between 500 and 985 samples, providing a relatively balanced evaluation setting across different construction defect types.

Each image is provided with three types of annotations:

- **Bounding Boxes:** Identify the location of the defect region, supporting object detection and detection-guided prompting.
- **Segmentation Masks:** Provide pixel-level ground-truth annotations for evaluating defect segmentation performance.
- **Class Labels:** Identify the corresponding construction defect category.

During dataset preparation, particular attention is paid to annotation consistency and visual diversity. The dataset covers variations in defect scale, illumination, surface materials, and background complexity commonly encountered in real-world construction environments.

The construction defect dataset can be organized into the following directory structure:

```text
Construction_Defect_Dataset/
├── Crack/
│   ├── images/
│   └── gt/
├── Damp_Stain/
│   ├── images/
│   └── gt/
├── Efflorescence/
│   ├── images/
│   └── gt/
├── Leakage_Water_Stain/
│   ├── images/
│   └── gt/
├── Rebar_Exposure/
│   ├── images/
│   └── gt/
├── Roof_Damage/
│   ├── images/
│   └── gt/
└── Spalling/
    ├── images/
    └── gt/
```

Each category contains the corresponding defect images and pixel-level ground-truth masks. Bounding-box annotations and class labels are also provided for detection-based evaluation.

The construction defect dataset is used exclusively as a **zero-shot evaluation benchmark** and is not involved in SAM fine-tuning.

Specifically:

- **Training Domain:** Industrial defect datasets are used to fine-tune SAM and learn generic defect representations.
- **Evaluation Domain:** Construction defect datasets are used to evaluate the transferability of the fine-tuned model to previously unseen construction defect categories.
- **Zero-Shot Setting:** No construction defect images or annotations are used during SAM fine-tuning.

This evaluation protocol aims to assess the generalization capability of SAM across different defect domains, reflecting practical construction inspection scenarios where pixel-level annotated data are limited.
