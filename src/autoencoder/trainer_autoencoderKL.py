import argparse
import yaml
import torch
from autoencoder import AutoencoderKL
import CustomModelCheckpoint as cmc
from dataset3d_npy import Dataset3D_NPY
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer



if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser(description='Train a VAE')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')

    # parse config
    args = parser.parse_args()
    config_filepath = args.config

    """
    Determine if any GPUs are available
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load config file
    with open(config_filepath, 'r') as config_file:
        configs = yaml.safe_load(config_file)

    ddconfig = configs['ddconfig']
    lossconfig = configs['lossconfig']
    hyperparams = configs['hyperparams']
    dataconf = configs['dataconf']
    modelconf = configs['model_restore']

    learning_rate = hyperparams['lr']
    batch_size = hyperparams['batch_size']
    num_epochs = hyperparams['epochs']
    embed_dim = hyperparams['embed_dim']

    normalize_mask = ddconfig['in_channels'] == ddconfig['out_ch']
    
    img_dir   = dataconf['train_img_dir']
    label_dir = dataconf['train_label_dir']

    val_img_dir   = dataconf['val_img_dir']
    val_label_dir = dataconf['val_label_dir']

    train_ds = Dataset3D_NPY(img_dir, label_dir)
    val_ds = Dataset3D_NPY(val_img_dir, val_label_dir)

    CKPT_PATH = modelconf['ckpt_path']
    if CKPT_PATH is None:
        print("Training from scratch...")
    else:
        print(f"Loading model from {CKPT_PATH}")

    network = AutoencoderKL(ddconfig, lossconfig, embed_dim, train_ds, 
        val_ds, batch_size, 
        ckpt_path=CKPT_PATH,
        learning_rate=learning_rate).to(device)

    # Init ModelCheckpoint callback, monitoring 'val_loss'
    checkpoint_callback = ModelCheckpoint(
        monitor="val/rec_loss",
        save_top_k=1, 
        filename="{epoch:04}"
    )

    custom_callback = cmc.CustomModelCheckpoint(
        config_filename=config_filepath
    )

    trainer = Trainer(
        max_epochs=num_epochs,
        progress_bar_refresh_rate=1,
        devices = 1,
        accelerator = "gpu",
        callbacks=[checkpoint_callback, custom_callback]
    )
    # Train the Model
    trainer.fit(network)

    print("Training done!")

