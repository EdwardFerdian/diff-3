
import os
import shutil
from pytorch_lightning.callbacks import ModelCheckpoint

class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, config_filename, **kwargs):
        self.config_filename = config_filename
        super().__init__(**kwargs)

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)

        print("\nCopying config file...\n")
        source_yaml = self.config_filename
        parent_dir = os.path.dirname(self.dirpath)
        target_yaml = os.path.join(parent_dir, 'vae_params.yaml')
        shutil.copyfile(source_yaml, target_yaml)