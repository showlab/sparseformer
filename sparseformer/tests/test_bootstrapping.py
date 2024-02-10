import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
from sparseformer.modelling import SparseFormer
from sparseformer.ckpt_utils import convert_ckpt_btsp_to_v2
from sparseformer.config import base_btsp_config
from PIL import Image
import torchvision.transforms as T


if __name__ == "__main__":

    im = Image.open("tench.jpeg")
    preprocess = T.Compose(
        [
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    x = preprocess(im).unsqueeze(0).repeat(2, 1, 1, 1)
    print(x.shape)

    net = SparseFormer(
        **base_btsp_config()
    )

    net.eval()

    state_dict = torch.load("/Users/sebgao/augreg_base_patch16_224.pth", map_location="cpu")
    state_dict = convert_ckpt_btsp_to_v2(state_dict)

    print(net.load_state_dict(state_dict, strict=False))

    print(net(x).argmax(1))


    IMG_SIZE = 224
    input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    flops = FlopCountAnalysis(net, input)
    print(flop_count_table(flops, max_depth=2))
