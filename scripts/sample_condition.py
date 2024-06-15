import os
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from diffusionlib.conditioning_method.torch import get_conditioning_method
from diffusionlib.dataset import get_dataset
from diffusionlib.config_definition.model import ModelConfig, ModelName
from diffusionlib.model.torch import create_model
from diffusionlib.noise import get_noise
from diffusionlib.operator import get_operator
from diffusionlib.sampler.torch import create_sampler
from diffusionlib.util.array import to_numpy
from diffusionlib.util.image import mask_generator
from diffusionlib.util.logger import get_logger


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    save_dir = Path() / "results"

    # logger
    logger = get_logger()

    # Device setting
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)

    # Load configurations
    model_config = ModelConfig.load(ModelName.FFHQ)
    diffusion_config = load_yaml(str((Path() / "config" / "diffusion.yaml").absolute()))
    task_config = load_yaml(
        str((Path() / "config" / "operator" / "super_resolution.yaml").absolute())
    )

    # Load model
    model = create_model(model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config["measurement"]
    operator = get_operator(device=device, **measure_config["operator"])
    noiser = get_noise(**measure_config["noise"])
    logger.info(
        f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}"
    )

    # Prepare dps conditioning method
    # dps_cond_config = task_config["dpsconditioning"]
    # dps_cond_method = get_conditioning_method(
    #     dps_cond_config["method"], operator, noiser, **dps_cond_config["params"]
    # )
    # dps_measurement_cond_fn = dps_cond_method.conditioning
    # logger.info(f"Conditioning method : {task_config['dpsconditioning']['method']}")

    # Prepare tmpd condition method
    tmpd_cond_config = task_config["tmpdconditioning"]
    tmpd_cond_method = get_conditioning_method(tmpd_cond_config["method"], operator, noiser)
    tmpd_measurement_cond_fn = tmpd_cond_method.conditioning
    logger.info(f"tmpd Conditioning method : {task_config['tmpdconditioning']['method']}")

    # Prepare pigdm condition method
    pigdm_cond_config = task_config["pigdmconditioning"]
    pigdm_cond_method = get_conditioning_method(pigdm_cond_config["method"], operator, noiser)
    pigdm_measurement_cond_fn = pigdm_cond_method.conditioning
    logger.info(f"pigdm Conditioning method : {task_config['pigdmconditioning']['method']}")

    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config)

    # sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
    # dps_sample_fn = partial(
    #     sampler.p_sample_loop,
    #     config=FLAGS.config,
    #     model=model,
    #     measurement_cond_fn=dps_measurement_cond_fn,
    # )
    pigdm_sample_fn = partial(
        # sampler.tmpd_sample_loop,
        sampler.pigdm_sample_loop,
        config=measure_config,
        model=model,
        measurement_cond_fn=pigdm_measurement_cond_fn,
    )
    tmpd_sample_fn = partial(
        sampler.tmpd_sample_loop,
        config=measure_config,
        model=model,
        measurement_cond_fn=tmpd_measurement_cond_fn,
    )

    # out_path = os.path.join(FLAGS.save_dir, measure_config['operator']['name'], 'deleteme')
    # out_path = os.path.join(FLAGS.save_dir, measure_config['operator']['name'] + measure_config['mask_opt']['mask_type'])  # for random mask
    out_path = os.path.join(
        save_dir,
        measure_config["operator"]["name"] + str(measure_config["operator"]["scale_factor"]),
    )  # for superresolution 8x
    # out_path = os.path.join(FLAGS.save_dir, measure_config['operator']['name'])
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ["input", "dps", "pigdm", "tmpd", "label"]:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)
        for noise in ["0.01", "0.05", "0.1", "0.2"]:
            os.makedirs(os.path.join(out_path, img_dir, noise), exist_ok=True)

    os.makedirs(os.path.join(out_path, "progress"), exist_ok=True)

    # Prepare TF dataloader
    eval_ds = get_dataset("ffhq").create_dataset(1)

    # Exception) In case of inpainting, we need to generate a mask
    if measure_config["operator"]["name"] == "inpainting":
        mask_gen = mask_generator(**measure_config["mask_opt"])

    # Do Inference
    for i, batch in enumerate(iter(eval_ds)):
        if i == 1000:
            assert 0  # break at 1k to evaluate FID-1k
        ref_img = batch["image"][0]
        # Convert to torch.Tensor
        ref_img = torch.Tensor(np.array(ref_img).transpose(0, 3, 1, 2)).to(device="cuda:0")

        # Exception) In case of inpainging,
        if measure_config["operator"]["name"] == "inpainting":
            mask = mask_gen(ref_img)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)

            # dps_measurement_cond_fn = partial(dps_cond_method.conditioning, mask=mask)
            # dps_sample_fn = partial(dps_sample_fn, measurement_cond_fn=dps_measurement_cond_fn)

            pigdm_measurement_cond_fn = partial(pigdm_cond_method.conditioning, mask=mask)
            pigdm_sample_fn = partial(
                pigdm_sample_fn, measurement_cond_fn=pigdm_measurement_cond_fn
            )

            tmpd_measurement_cond_fn = partial(tmpd_cond_method.conditioning, mask=mask)
            tmpd_sample_fn = partial(tmpd_sample_fn, measurement_cond_fn=tmpd_measurement_cond_fn)

            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img, mask=mask)
            y_n = noiser(y)

        else:
            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img)
            y_n = noiser(y)

        # Sampling
        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()

        fname = str(i).zfill(5) + ".png"
        noise = str(measure_config["noise"]["sigma"])
        plt.imsave(os.path.join(out_path, "input", noise, fname), to_numpy(y_n))
        plt.imsave(os.path.join(out_path, "label", noise, fname), to_numpy(ref_img))

        # dps_sample = dps_sample_fn(
        #     x_start=x_start, measurement=y_n, record=True, save_root=out_path
        # )
        # plt.imsave(os.path.join(out_path, "dps", noise, fname), to_numpy(dps_sample))

        pigdm_sample = pigdm_sample_fn(
            x_start=x_start, measurement=y_n, record=True, save_root=out_path
        )
        plt.imsave(os.path.join(out_path, "pigdm", noise, fname), to_numpy(pigdm_sample))

        tmpd_sample = tmpd_sample_fn(
            x_start=x_start, measurement=y_n, record=True, save_root=out_path
        )
        plt.imsave(os.path.join(out_path, "tmpd", noise, fname), to_numpy(tmpd_sample))


if __name__ == "__main__":
    main()
