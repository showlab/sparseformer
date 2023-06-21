# Image SparseFormer

## Result
| config                                    | ImageNet top-1 | GFLOPs | Params | ckpt   | log |
|-------------------------------------------|----------------|--------|--------|--------|-----|
| [SF-Tiny](./configs/sparseformer_t.yaml)  | 81.0           | 2.0G   | 32M    | [(google drive)](https://drive.google.com/file/d/1ldsK-8ZaJ0vz1uzGzXAmJGnAwvTe0Kcl/view?usp=drive_link) | [(google drive)](https://drive.google.com/file/d/1ldsK-8ZaJ0vz1uzGzXAmJGnAwvTe0Kcl/view?usp=drive_link) |
| [SF-Small](./configs/sparseformer_s.yaml) | 82.0           | 3.8G   | 48M    | [(google drive)](https://drive.google.com/file/d/1YqzEeMWdg9VQUunj_d6M0NWlTqfdxRC2/view?usp=drive_link) | [(google drive)](https://drive.google.com/file/d/1YqzEeMWdg9VQUunj_d6M0NWlTqfdxRC2/view?usp=drive_link) |
| [SF-Base](./configs/sparseformer_b.yaml)  | 82.6           | 7.8G   | 81M    | [(google drive)](https://drive.google.com/file/d/1Ko_lBXnX_fWDh5b9lkwEEvYIt11Q5pOD/view?usp=drive_link) | [(google drive)](https://drive.google.com/file/d/1Ko_lBXnX_fWDh5b9lkwEEvYIt11Q5pOD/view?usp=drive_link) |
## Main model file
It is in [models/sparseformer.py](./models/sparseformer.py).
## Usage
### Training
You can start training our SparseFormer on ImageNet-1K with make targets in [Makefile](./Makefile) when the environment is installed (see the following).

### AMP Caveats
Now we support automatic mixed precision (AMP) training with `torch.autocast`! Specify AMP_OPT_LEVEL=A to enable it. In case of your own training loop, please pass grad scale into the forward method of Sparseformer, like `outputs = model(samples, scale=scaler.get_scale())`. We have tested it on SparseFormer-T and the accuracy is matched.

~~Till now, we have not yet support mixed precision training of SparseFormer models but it is a planned feature.~~

### Install
(based on the original Swin [getting_started.md](https://github.com/microsoft/Swin-Transformer/blob/main/get_started.md))

- Clone this repo:

```bash
git clone https://github.com/showlab/sparseformer.git
cd sparseformer
```

- Create a conda virtual environment and activate it:

```bash
conda create -n sf python=3.7 -y
conda activate sf
```

- Install `CUDA==10.1` with `cudnn7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch==1.7.1` and `torchvision==0.8.2` with `CUDA==10.1`:

```bash
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
```

- Install `timm==0.3.2`:

```bash
pip install timm==0.3.2
```

- Install other requirements:

```bash
pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8
```

### Data preparation

We use standard ImageNet dataset, you can download it from http://image-net.org/. We provide the following two ways to
load data:

- For standard folder dataset, move validation images to labeled sub-folders. The file structure should look like:
  ```bash
  $ tree data
  imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
 
  ```
- To boost the slow speed when reading images from massive small files, we also support zipped ImageNet, which includes
  four files:
    - `train.zip`, `val.zip`: which store the zipped folder for train and validate splits.
    - `train_map.txt`, `val_map.txt`: which store the relative path in the corresponding zip file and ground truth
      label. Make sure the data folder looks like this:

  ```bash
  $ tree data
  data
  └── ImageNet-Zip
      ├── train_map.txt
      ├── train.zip
      ├── val_map.txt
      └── val.zip
  
  $ head -n 5 data/ImageNet-Zip/val_map.txt
  ILSVRC2012_val_00000001.JPEG	65
  ILSVRC2012_val_00000002.JPEG	970
  ILSVRC2012_val_00000003.JPEG	230
  ILSVRC2012_val_00000004.JPEG	809
  ILSVRC2012_val_00000005.JPEG	516
  
  $ head -n 5 data/ImageNet-Zip/train_map.txt
  n01440764/n01440764_10026.JPEG	0
  n01440764/n01440764_10027.JPEG	0
  n01440764/n01440764_10029.JPEG	0
  n01440764/n01440764_10040.JPEG	0
  n01440764/n01440764_10042.JPEG	0
  ```

