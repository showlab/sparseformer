import torch
import urllib.request 
import os

import torchvision.transforms as T

from .config import route_name_to_config
from .modelling import SparseFormer


MODEL_CKPT_URLS = {
    # SparseFormer: Sparse Visual Recognition via Limited Latent Tokens
    "sparseformer_v1_tiny": "https://github.com/showlab/sparseformer/releases/download/v2.0.0-refactor/sparseformer_v1_tiny_converted.pth",
    "sparseformer_v1_small": "https://github.com/showlab/sparseformer/releases/download/v2.0.0-refactor/sparseformer_v1_small_converted.pth",
    "sparseformer_v1_base": "https://github.com/showlab/sparseformer/releases/download/v2.0.0-refactor/sparseformer_v1_base_converted.pth",
    # Bootstrapping SparseFormers from Vision Foundation Models
    # unimodal augreg ones
    "sparseformer_btsp_augreg_base": "https://github.com/showlab/sparseformer/releases/download/v2.0.0-refactor/sparseformer_btsp_augreg_base_converted.pth",
    "sparseformer_btsp_augreg_large": "https://github.com/showlab/sparseformer/releases/download/v2.0.0-refactor/sparseformer_btsp_augreg_large_converted.pth",
    # multimodal openai clip ones
    "sparseformer_btsp_openai_clip_base": "https://github.com/showlab/sparseformer/releases/download/v2.0.0-refactor/sparseformer_btsp_openai_clip_base_converted.pth",
    "sparseformer_btsp_openai_clip_large": "https://github.com/showlab/sparseformer/releases/download/v2.0.0-refactor/sparseformer_btsp_openai_clip_large_converted.pth",
}


def create_preprocessing(resize, crop, mean, std):
    return T.Compose(
        [
            T.Resize(resize),
            T.CenterCrop(crop),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def create_model(model_type: str, pretrained=None, download=False, cache_dir="./ckpts/", return_only_model=True):
    model_type = model_type.lower()
    config = route_name_to_config(model_type)

    if download:
        assert model_type in MODEL_CKPT_URLS
        os.makedirs(cache_dir, exist_ok=True)
        pretrained = f"{cache_dir}/{model_type}.pth"
        if not os.path.exists(pretrained):
            urllib.request.urlretrieve(MODEL_CKPT_URLS[model_type], pretrained)

    if pretrained:
        state_dict = torch.load(pretrained, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]

    model = SparseFormer(**config)
    model.eval()

    print(model.load_state_dict(state_dict, strict=False))
    
    if return_only_model:
        return model
    else:
        return model,  create_preprocessing(256, 224, config["normalization_factor"][0], config["normalization_factor"][1])
