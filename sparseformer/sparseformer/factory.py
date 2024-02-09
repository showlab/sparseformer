import torch

from .config import route_name_to_config
from .modelling import SparseFormer

def create_model(model_type, pretrained):
    config = route_name_to_config(model_type)

    state_dict = torch.load(pretrained, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]

    config.pop("model_type")

    model = SparseFormer(**config)
    model.eval()

    print(model.load_state_dict(state_dict, strict=False))

    return model