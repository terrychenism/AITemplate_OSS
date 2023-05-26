#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import torch

from aitemplate.compiler import compile_model
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target

from ..modeling.unet_2d_condition import (
    UNet2DConditionModel as ait_UNet2DConditionModel,
)
from .util import mark_output

from ..pipeline_stable_diffusion_ait_alt import convert_ldm_unet_checkpoint

def map_unet_params(pt_mod, dim):
    pt_params = dict(pt_mod.named_parameters())
    params_ait = {}
    for key, arr in pt_params.items():
        if len(arr.shape) == 4:
            arr = arr.permute((0, 2, 3, 1)).contiguous()
        elif key.endswith("ff.net.0.proj.weight"):
            w1, w2 = arr.chunk(2, dim=0)
            params_ait[key.replace(".", "_")] = w1
            params_ait[key.replace(".", "_").replace("proj", "gate")] = w2
            continue
        elif key.endswith("ff.net.0.proj.bias"):
            w1, w2 = arr.chunk(2, dim=0)
            params_ait[key.replace(".", "_")] = w1
            params_ait[key.replace(".", "_").replace("proj", "gate")] = w2
            continue
        params_ait[key.replace(".", "_")] = arr

    params_ait["arange"] = (
        torch.arange(start=0, end=dim // 2, dtype=torch.float32).cuda().half()
    )
    return params_ait


def compile_unet(
    pt_mod,
    batch_size=2,
    height=64,
    width=64,
    dim=320,
    hidden_dim=1024,
    use_fp16_acc=False,
    convert_conv_to_gemm=False,
    attention_head_dim=[5, 10, 20, 20],  # noqa: B006
    model_name="UNet2DConditionModel",
    use_linear_projection=False,
):
    block_out_channels = [384, 768, 1536, 1536]
    ait_mod = ait_UNet2DConditionModel(
        sample_size=64,
        cross_attention_dim=hidden_dim,
        attention_head_dim=attention_head_dim,
        use_linear_projection=use_linear_projection,
        block_out_channels=block_out_channels,
        layers_per_block=3,
    )
    ait_mod.name_parameter_tensor()


    ait_param_names = [x[0] for x in ait_mod.named_parameters()]
    # print(ait_param_names)

    # set AIT parameters
    state_dict = torch.load("/home/terrychen/projects/model_0006999.pth", map_location="cpu")["model"]
    unet_state_dict = {}
    for key in state_dict.keys():
        if key.startswith("model.model.diffusion_model."):
            new_key = key.replace("model.model.diffusion_model.", "")
            unet_state_dict[new_key] = state_dict[key]


    pt_param_names = [k for k, _ in unet_state_dict.items()]
    # print(pt_param_names)

    unet_state_dict = convert_ldm_unet_checkpoint(unet_state_dict, layers_per_block=3)

    # params_ait = map_unet_params(pt_mod, block_out_channels[0])
    params_ait = unet_state_dict

    # batch_size = IntVar(values=[1, 8], name="batch_size")
    height_d = 64 #IntVar(values=[32, 64], name="height")
    width_d = 64 #IntVar(values=[32, 64], name="width")

    latent_model_input_ait = Tensor(
        [batch_size, height_d, width_d, 4], name="input0", is_input=True
    )
    timesteps_ait = Tensor([batch_size], name="input1", is_input=True)
    text_embeddings_pt_ait = Tensor(
        [batch_size, 77, hidden_dim], name="input2", is_input=True
    )
    t5_embeddings_pt_ait = Tensor(
        [batch_size, 77, 2048], name="input3", is_input=True
    )

    mid_block_additional_residual = None
    down_block_additional_residuals = None

    Y = ait_mod(
        latent_model_input_ait,
        timesteps_ait,
        [text_embeddings_pt_ait, t5_embeddings_pt_ait],
        down_block_additional_residuals,
        mid_block_additional_residual,
    )
    mark_output(Y)

    target = detect_target(
        use_fp16_acc=use_fp16_acc, convert_conv_to_gemm=convert_conv_to_gemm
    )
    compile_model(Y, target, "./tmp", model_name, constants=params_ait)
