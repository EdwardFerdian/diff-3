from tqdm import tqdm
import os
from multiprocessing import cpu_count
import torch
import yaml
from autoencoder import AutoencoderKL
from dataset3d_npy import Dataset3D_NPY
from torch.utils.data import DataLoader
import numpy as np
import util.h5_util as h5util
import argparse

if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser(description='Test a VAE')
    parser.add_argument('--subset', type=str, default="train", help='train/val')
    parser.add_argument('--model-dir', type=str, help='Path to the models dir', required=True)
    parser.add_argument('--output-dir', type=str, help='Path to the output dir', required=True)
    parser.add_argument('--version', type=str, help='Version nr of the model/data')
    
    args = parser.parse_args()
    subset = args.subset
    model_dir = args.model_dir
    output_dir = args.output_dir
    ver = args.version
        
    # prepare output dir
    if ver is not None:
        output_path = f"{output_dir}/latent_{ver}.h5"
    else:
        output_path = f"{output_dir}/latent.h5"

    # check exist create output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    version_dir = f"{model_dir}/checkpoints" 
    config_filepath = f"{model_dir}/vae_params.yaml"

    # config stuff
    with open(config_filepath, 'r') as config_file:
        configs = yaml.safe_load(config_file)

    ddconfig = configs['ddconfig']
    lossconfig = configs['lossconfig']
    hyperparams = configs['hyperparams']
    dataconf = configs['dataconf']
    
    embed_dim = hyperparams['embed_dim']

    img_dir   = dataconf['train_img_dir']
    label_dir = dataconf['train_label_dir']

    val_img_dir   = dataconf['val_img_dir']
    val_label_dir = dataconf['val_label_dir']

    # load dataset
    train_ds = Dataset3D_NPY(img_dir, label_dir)
    val_ds = Dataset3D_NPY(val_img_dir, val_label_dir)
    
    # choose which dataset to use
    ds = train_ds if subset == "train" else val_ds 


    network = AutoencoderKL(ddconfig, lossconfig, embed_dim, train_ds, val_ds, batch_size=1).to(device)
    # summary(network, (2, 160, 160, 128))

    # get last ckpt file
    ckpt_files = os.listdir(version_dir)
    ckpt_files = [f for f in ckpt_files if f.startswith("epoch=0")]
    # sort and take last one
    ckpt_files.sort()
    CKPT_PATH = rf"{version_dir}/{ckpt_files[-1]}"
    print(f"Loading model from {CKPT_PATH}")
    
    

    # load model checkpoint
    checkpoint = torch.load(CKPT_PATH)
    print(f"loading from checkpoint  {CKPT_PATH}")
    print(checkpoint.keys())

    network.load_state_dict(checkpoint['state_dict'], strict=False)
    network.eval()

    dl = DataLoader(ds, batch_size=1,shuffle=False, pin_memory = True, num_workers = cpu_count())
    
    with torch.no_grad():
        for data in tqdm(dl):
            imgs = data
            imgs = imgs.to(device)
            
            posterior = network.encode(imgs)
            mu = posterior.mean.cpu().numpy()
            std = posterior.std.cpu().numpy()
            
            # for each of the mu-std pair, calculate the lower/upper bound and get the min/max
            lower_bound = np.min(mu - 3 * std, axis=(1,2,3,4))
            upper_bound = np.max(mu + 3 * std, axis=(1,2,3,4))

            h5util.save(output_path, f"{subset}/mean", mu)
            h5util.save(output_path, f"{subset}/std", std)
            
            h5util.save(output_path, f"{subset}/min", lower_bound)
            h5util.save(output_path, f"{subset}/max", upper_bound)
    
    print(f"Saved latent space to {output_path}")


