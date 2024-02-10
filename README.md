# 🎆 SparseFormer
This is the offical repo for SparseFormer researches:

> [**SparseFormer: Sparse Visual Recognition via Limited Latent Tokens**](https://arxiv.org/abs/2304.03768) **(ICLR 2024)**<br>
> Ziteng Gao, Zhan Tong, Limin Wang, Mike Zheng Shou<br>

> [**Bootstrapping SparseFormers from Vision Foundation Models**](https://arxiv.org/abs/2312.01987)<br>
> Ziteng Gao, Zhan Tong, Kevin Qinghong Lin, Joya Chen, Mike Zheng Shou<br>


## Out-of-box SparseFormer as a Library (recommended)
We provide the out-of-box SparseFormer usage including both original and bootstrapped ones [here](./sparseformer/).

You can install sparseformer as a library by the following command:
```shell
pip install -e sparseformer # in this folder
```

then you can use SparseFormers out of box in Python:
```python
from sparseformer.factory import create_model

# e.g., make a SparseFormer v1 tiny model
model = create_model("sparseformer_v1_tiny", download=True)


# or make a Clip SparseFormer large model and put it in OpenClip pipeline
import open_clip
clip = open_clip.create_model_and_transforms("ViT-L-14", "openai")
visual = create_model("sparseformer_openai_clip_large", download=True)
clip.visual = visual
# ...

```

More detailed documents will be available soon.



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