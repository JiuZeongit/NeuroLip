# NeuroLip

This is an official implementation of [NeuroLip: An Event-driven Spatiotemporal Learning Framework for Cross-Scene Lip-Motion-based Visual Speaker Recognition](doi link)
---

## Installation

This project requires **Python >= 3.10** and the following packages:

- matplotlib >= 3.10.7
- numpy >= 2.2.6
- opencv-python >= 4.12.0.88
- scikit-learn >= 1.7.2
- tensorboard >= 2.20.0
- torch >= 2.8.0
- torchvision >= 0.23.0
- tqdm >= 4.67.1

### 1) Create a Conda virtual environment

```bash
conda create -n neurolip python=3.10 -y
```

### 2) Activate the environment

```bash
conda activate neurolip
```

### 3) Install PyTorch and TorchVision

> **Important:** Install PyTorch first, and choose the command that matches your CUDA version.  
> If you are not sure, use the CPU version first, or check the official PyTorch installation page.

#### Option A: CUDA (recommended if you have an NVIDIA GPU)

Example (CUDA 12.8):

```bash
pip install "torch>=2.8.0" "torchvision>=0.23.0" --index-url https://download.pytorch.org/whl/cu128
```

#### Option B: CPU only

```bash
pip install "torch>=2.8.0" "torchvision>=0.23.0" --index-url https://download.pytorch.org/whl/cpu
```

### 4) Install the remaining dependencies

```bash
pip install \
  "matplotlib>=3.10.7" \
  "numpy>=2.2.6" \
  "opencv-python>=4.12.0.88" \
  "scikit-learn>=1.7.2" \
  "tensorboard>=2.20.0" \
  "tqdm>=4.67.1"
```

### 5) Verify the installation

```bash
python - <<'PY'
import sys
import torch
import torchvision
import numpy
import cv2
import sklearn
import matplotlib
import tqdm
import tensorboard

print("Python:", sys.version)
print("Torch:", torch.__version__)
print("TorchVision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
print("NumPy:", numpy.__version__)
print("OpenCV:", cv2.__version__)
print("scikit-learn:", sklearn.__version__)
print("matplotlib:", matplotlib.__version__)
print("tqdm:", tqdm.__version__)
print("TensorBoard:", tensorboard.__version__)
PY
```

---

## Dataset Format

### Toy dataset

You can check this repo by using the toy dataset in fold ToyDataset, and this toy dataset is a subset of DVSpeaker dataset.
If you need the complete DVSpeaker dataset, you can refer to the next section "DVSpeaker Dataset".

### DVSpeaker dataset

**Acquisition Setup:** DVSpeaker was collected from **50 participants**.

DVSpeaker was recorded by a **Prophesee EVK4 event camera** (1280×720) and a **Logitech C922 RGB camera** (1920×1080, 60 FPS). Participants sat **90–110 cm** from the cameras and articulated randomly presented digits (**0–9**). For personal privacy, the RGB data will not be available.

Data were collected under **four acquisition conditions** to cover viewpoint and illumination variations:
- **SI-0°**, **SI-45°**, **SI-90°** (sufficient illumination, ~216 lux)
- **II-0°** (insufficient illumination, ~12.5 lux)

For each condition, we collected **100 valid samples per participant**.


**Dataset Description:** 

You can download DVSpeaker by link (dataset link)

DVSpeaker folder layout:

```text
DVSpeaker/
├── 1/
│   ├── 1_0_0_0_0.npy
│   ├── 1_45_3_1_2.npy
│   └── ...
├── 2/
│   └── ...
...
└── 50/
```

Filename format:

```text
{light}_{degree}_{num}_{A}_{B}.npy
```

Inside DVSpeaker dataset, there are **50** subfolders named from **`1`** to **`50`**, and these folder names correspond to the ID labels. Each subfolder contains multiple **`.npy`** files. Each **`.npy`** file stores a array with shape **`(N,)`**, where each element is a structured event record containing **`x`**, **`y`**, **`p`**, and **`t`**. The filename of each **`.npy`** file follows the format **`{light}_{degree}_{num}_{A}_{B}.npy`**. Here, **`light`** takes two illumination values (**`0`**(Insufficient illumination, ~12.5 lux) or **`1`**(Sufficient Illumination, ~216 lus)), **`degree`** takes three shooting degree (**`0`**, **`45`**, **`90`**), and **`num`** takes ten values (**`0–9`**) corresponding to the speaking content. A and B have no significance; they are merely used to differentiate files.


**Access to the DVSpeaker dataset:** 
Download the [DVSpeaker Dataset Release Agreement](https://github.com/JiuZeongit/NeuroLip/blob/main/ToyDataset/DVSpeakerDatasetReleaseAgreement.pdf) then fill in it to send to contact email:  [zhengyue@cuhk.edu.cn](zhengyue@cuhk.edu.cn)

---

## Train

Example: train on SI-0°, test on SI-45°， Use toy dataset

```bash
python main.py \
  --data_root  ToyDataset \
  --log_dir ./log \
  --train_light 1 \
  --train_degree 0 \
  --test_light 1 \
  --test_degree 45 \
  --batch_size 8 \
  --num_epochs 30 \
  --device cuda:0 
```

### Notes
- Validation is split from the training scene (80/20).
- Best checkpoint is selected by validation loss.
- Validation is from the training data so as to simulate the situation where it is impossible to obtain data from the target domain in real life.
---

## Test

Evaluate a saved checkpoint on a target scene:

```bash
python test.py \
  --data_root ToyDataset \
  --checkpoint ./log/model_best.pth \
  --test_light 1 \
  --test_degree 45 \
  --batch_size 8 \
  --device cuda:0
```

Additionally, if you want to reproduce the results reported in the NeuroLip paper, you can use **NeuroLipTrained.pth** in log fould for testing.


## Citation

If you find our work useful in your research, please cite:

```
@InProceedings{
}
```
