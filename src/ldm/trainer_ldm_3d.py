
import argparse
from ldm.denoising_diffusion_pytorch3D import Unet3D, GaussianDiffusion3D, Trainer3D

if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser(description='Train LDM')
    parser.add_argument('--ver', type=str, required=True, help='Version nr of the model')
    parser.add_argument('--input', type=str, help='Path to H5 file input', required=True)
    parser.add_argument('--output-dir', type=str, help='Path to output dir', required=True)
    parser.add_argument('--batch-size', type=int, help='Batch size', default=64)
    args = parser.parse_args()

    ver = args.ver
    h5_filepath = args.input
    output_dir = args.output_dir
    batch_size = args.batch_size

    output_dir = f"{output_dir}/diff3_model_{ver}"
    
    im_size = (20,20,16)
    channel = 4
    sample_step = 250
    

    model = Unet3D(
        dim = 64,
        dim_mults = (1, 2, 4),
        channels = channel
    ).cuda()

    diffusion = GaussianDiffusion3D(
        model,
        image_size = im_size,
        timesteps = 1000,           # number of steps
        sampling_timesteps = sample_step,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type = 'l1'            # L1 or L2
    ).cuda()

    trainer = Trainer3D(
        diffusion,
        h5_filepath = h5_filepath,
        # image_size = im_size,
        dynamic_sampling = True,
        train_batch_size = batch_size,
        train_lr = 8e-5,
        train_num_steps = 50000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        # amp = True                        # turn on mixed precision
        results_folder = output_dir,
        num_samples=25,
        save_and_sample_every = 1000,
    )

    trainer.train()

