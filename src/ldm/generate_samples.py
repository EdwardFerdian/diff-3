from denoising_diffusion_pytorch3D import Unet3D, GaussianDiffusion3D, Trainer3D
import torch
from torchvision import utils
import math
import numpy as np
import h5py
import argparse
import util.h5_util as h5util

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

if __name__ == "__main__":
    
    # args
    parser = argparse.ArgumentParser(description='Generate latent samples')
    parser.add_argument('--model-dir', type=str, required=True, help='Path to the models dir')
    parser.add_argument('--model-id', type=str, required=True, help='Model id of the model to test')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to the output dir')
    parser.add_argument('--version', type=str, help='Version nr of samples')
    parser.add_argument('--n-data', type=int, default=4, help='Number of data to generate')
    args = parser.parse_args()

    ver = args.version
    model_id = args.model_id
    num_samples = args.n_data
    ldm_network_dir = args.model_dir

    if ver is not None:
        output_filename = f"_samples{ver}.h5"
    else:
        output_filename = f"_samples.h5"
    
    im_size = (20,20,16)
    channel = 4
    sample_step = 100
    batch_size = 32

    model = Unet3D(
        dim = 64,
        dim_mults = (1, 2, 4),
        channels = channel
    )

    diffusion = GaussianDiffusion3D(
        model,
        image_size = im_size,
        sampling_timesteps = sample_step,
    )

    trainer = Trainer3D(
        diffusion,
        h5_filepath = None,
        results_folder = ldm_network_dir,
        num_samples=num_samples
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainer.load(model_id)
    trainer.ema.to(device)
    trainer.ema.ema_model.eval()

    with torch.no_grad():
        total_sample = 0
        batches = num_to_groups(trainer.num_samples, trainer.batch_size)
        for batch in batches:
            images = trainer.ema.ema_model.sample(batch_size=batch)
            h5util.save(f"{trainer.results_folder}/{output_filename}", f"iter-{model_id}/result", images.cpu().numpy(), compression="gzip")

            total_sample += batch
            print(f"Generated {total_sample}/ {num_samples}")
            
    # we need to get the global min/max to denormalize
    h5_filepath = f"{ldm_network_dir}/_result.h5" # this is the logging file from our LDM training
    with h5py.File(h5_filepath, "r") as f:
        global_min = f["min"][0]
        global_max = f["max"][0]
    h5util.save(f"{trainer.results_folder}/{output_filename}", "min", np.asarray([global_min]))
    h5util.save(f"{trainer.results_folder}/{output_filename}", "max", np.asarray([global_max]))

    print("Samples saved to: ", f"{trainer.results_folder}/{output_filename}")
