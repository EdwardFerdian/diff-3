import argparse
import os
import numpy as np
from  util.mhd_util import read_mhd
import util.h5_util as h5util
from scipy.ndimage import zoom



class Preprocessor():
    def __init__(
        self,
        data_dir,
        label_dir,
        image_size,
        output_dir,
        subset,
        format = "npy",
        ext = 'nii.gz'
    ):
        super().__init__()
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.new_shape=image_size
        self.output_dir = output_dir
        self.data_subset = subset
        assert format in ["npy", "h5"]
        self.format = format
        
        self.filenames = [p[:-(len(ext)+1)] for p in os.listdir(label_dir)]
        
        print(f'{len(self.filenames) = }')
        
        self.img_paths = []
        self.label_paths = []
        for p in self.filenames:
            if os.path.exists(f"{label_dir}/{p}.{ext}"):
                self.img_paths.append(f"{data_dir}/{p}_0000.{ext}")
                self.label_paths.append(f"{label_dir}/{p}.{ext}")

    def _calculate_ratio_pads(self, img):
         # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:3]  # current shape [height, width]
        
        # Scale ratio (new / old)
        r = min(self.new_shape[0] / shape[0], self.new_shape[1] / shape[1], self.new_shape[2] / shape[2])

        r = 1.0 if r >= 1.0 else r

        
        new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r)), int(round(shape[2] * r))
        dx = self.new_shape[0] - new_unpad[0]
        dy = self.new_shape[1] - new_unpad[1]
        dz = self.new_shape[2] - new_unpad[2]

        # divide padding into 2 sides
        lx, ly, lz = dx // 2, dy // 2, dz // 2
        rx, ry, rz = dx - lx, dy - ly, dz - lz

        pads = ((lx, rx), (ly, ry), (lz, rz))
        # print(r, pads, shape)
        return r, pads
    
    def _resize(self, img, r, pads, order):
        img = zoom(img, r, order=order)
        img = np.pad(img, pads)
        return img

    def process_data(self, index):
        path = self.img_paths[index]
        img = read_mhd(path)

        mask_path = self.label_paths[index]
        mask = read_mhd(mask_path)
        
        r, pads = self._calculate_ratio_pads(img)
        img = self._resize(img, r, pads, 3)
        mask = self._resize(mask, r, pads, 0)

        # as uint8
        img = img.astype(np.uint8)
        mask = mask.astype(np.uint8)

        if self.format == "h5":
            output_filepath = f"{self.output_dir}/mitea.h5"

            img = np.expand_dims(img, axis=0)
            mask = np.expand_dims(mask, axis=0)

            h5util.save(output_filepath, f"{self.data_subset}/images", img, dtype="uint8")
            h5util.save(output_filepath, f"{self.data_subset}/labels", mask, dtype="uint8")
            h5util.save_str(output_filepath, f"{self.data_subset}/filenames", self.filenames[index])
        else:
            img_dir = f"{self.output_dir}/{self.data_subset}/images"
            label_dir = f"{self.output_dir}/{self.data_subset}/labels"

            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(label_dir, exist_ok=True)
            
            np.save(f"{img_dir}/{self.filenames[index]}.npy", img)            
            np.save(f"{label_dir}/{self.filenames[index]}.npy", mask)
      
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--label-dir", type=str, required=True)
    parser.add_argument("--image-size", type=int, nargs="+", default=[160, 160, 128])
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--subset", type=str, required=True)
    parser.add_argument("--format", type=str, default="npy")
    args = parser.parse_args()

    data_dir = args.data_dir
    label_dir = args.label_dir
    image_size = args.image_size
    output_dir = args.output_dir
    subset = args.subset
    format = args.format

    prepro = Preprocessor(data_dir, label_dir, image_size, output_dir, subset, format)

    for i in range(len(prepro.filenames)):
        print(f"Processing image {i+1}/{len(prepro.filenames)}", end="\r")
        prepro.process_data(i)