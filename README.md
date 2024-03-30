# 🎆 SparseFormer

This is the offical repo for SparseFormer researches:

> [**SparseFormer: Sparse Visual Recognition via Limited Latent Tokens**](https://arxiv.org/abs/2304.03768) **(ICLR 2024)**<br>
> Ziteng Gao, Zhan Tong, Limin Wang, Mike Zheng Shou<br>

> [**Bootstrapping SparseFormers from Vision Foundation Models**](https://arxiv.org/abs/2312.01987) **(CVPR 2024)**<br>
> Ziteng Gao, Zhan Tong, Kevin Qinghong Lin, Joya Chen, Mike Zheng Shou<br>

<!-- ## TL;DR
SparseFormer is a ViT with **less tokens and compute used**, which can also handle **any aspect ratio and resolution**. -->

## Out-of-box SparseFormer as a Library (recommended)
We provide the out-of-box SparseFormer usage with the sparseformer library installation. 

__Getting started__. You can install sparseformer as a library by the following command:
```shell
pip install -e sparseformer # in this folder
```

Available pre-trained model weights are listed [here](./sparseformer/sparseformer/factory.py#L11), including weights of v1 and bootstrapped ones. You can simply use [`create_model`](./sparseformer/sparseformer/factory.py#L37) with the argument `download=True` to get pre-trained models. You can play like this!
```python
from sparseformer.factory import create_model

# e.g., make a SparseFormer v1 tiny model
model = create_model("sparseformer_v1_tiny", download=True)


# or make a CLIP SparseFormer large model and put it in OpenClip pipeline
import open_clip
clip = open_clip.create_model_and_transforms("ViT-L-14", "openai")
visual = create_model("sparseformer_btsp_openai_clip_large", download=True)
clip.visual = visual
# ...

```

__Video SparseFormers__. We also provide unified [`MediaSparseFormer`](./sparseformer/sparseformer/media.py#L103) implementation for both video and image inputs (an image as single-frame video) with the token inflation argument `replicates`. MediaSparseFormer can load pre-trained weights of the image `SparseFormer` by [`load_2d_state_dict`](./sparseformer/sparseformer/media.py#L147).

Notes: Pre-trained weights VideoSparseFormers are currently unavailable. We might reproduce VideoSparseFormers if highly needed by the community.

__ADVANCED: Make your own SparseFormer and load timm weights__. 
Our codebase is generally compatible with [timm vision transformer](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py) weights. So here comes something to play: you can make your own SparseFormer and load timm transformers weights, not limited to our provided configurations!

For example, you can make a SparseFormer similar to ViT-224/16 and with sampling & decoding and roi adjusting every 3 block, and load it with CLIP OpenAI official pre-trained weights:
```python
from sparseformer.modeling import SparseFormer, OP
from sparseformer.config import base_btsp_config

ops_list = []
num_layers = 12
for i in range(num_layers):
    if i % 3 == 0:
        ops_list.append([OP.SAMPLING_M, OP.ATTN, OP.MLP, OP.ROI_ADJ, OP.PE_INJECT,])
    else:
        ops_list.append([OP.ATTN, OP.MLP])

config = base_btsp_config()
config.update(
    num_latent_tokens=16,
    num_sampling_points=9,
    width_configs=[768, ]*num_layers,
    repeats=[1, ]*num_layers,
    ops_list=ops_list,
)

model = SparseFormer(**config)

import timm
pretrained = timm.create_model("vit_base_patch16_clip_224.openai", pretrained=True)
new_dict = dict()
old_dict = pretrained.state_dict()
for k in old_dict:
    nk = k
    if "blocks" in k:
        nk = nk.replace("blocks", "layers")
    new_dict[nk] = old_dict[k]
print(model.load_state_dict(new_dict, strict=False))
```
All weights attention and MLP layers should be successfully loaded. The resulted SparseFormer should be fine-tuned to output meaningful results since the sampling & decoding and roi adjusting part are newly initialized. Maybe you can fine-tune it to be a CLIP-based open-vocabulary detector (have not yet tried, but very promising imo! :D).



## Training (SparseFormer v1)
For training SparseFormer v1 in ImageNets ([**SparseFormer: Sparse Visual Recognition via Limited Latent Tokens**](https://arxiv.org/abs/2304.03768)), please check [imagenet](./imagenet/).

**Note:** this [imagenet](./imagenet/) sub-codebase will be refactored soon.


## Citation
If you find SparseFormer useful in your research or work, please consider citing us using the following entry:
```
@misc{gao2023sparseformer,
    title={SparseFormer: Sparse Visual Recognition via Limited Latent Tokens},
    author={Ziteng Gao and Zhan Tong and Limin Wang and Mike Zheng Shou},
    year={2023},
    eprint={2304.03768},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@misc{gao2023bootstrapping,
      title={Bootstrapping SparseFormers from Vision Foundation Models}, 
      author={Ziteng Gao and Zhan Tong and Kevin Qinghong Lin and Joya Chen and Mike Zheng Shou},
      year={2023},
      eprint={2312.01987},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
