import h5py
import numpy as np

def load_img(filepath, col_name, idx):
    with h5py.File(filepath, 'r') as hl:
        img = hl.get(col_name)[idx]
        img = np.expand_dims(img, 0)
        
    return img

def save_str(output_filepath, col_name, str_val, compression=None):
    if not isinstance(str_val, list):
        str_val = [str_val]

    with h5py.File(output_filepath, 'a') as hf:    
        if col_name not in hf:
            ds = hf.create_dataset(col_name, shape=(len(str_val),), maxshape=(None,), dtype=h5py.string_dtype(encoding='utf-8'))
            # Write the list of strings to the dataset
            ds[:] = str_val
        else:
            ds = hf[col_name]
            ds.resize((ds.shape[0]) + len(str_val), axis = 0)
            ds[-len(str_val):] = str_val

def save(output_filepath, col_name, dataset, compression=None, dtype=None):
    if dtype is not None:
        dataset = np.array(dataset, dtype=dtype)

    # convert float64 to float32 to save space
    if dataset.dtype == 'float64':
        dataset = np.array(dataset, dtype='float32')
    
    with h5py.File(output_filepath, 'a') as hf:    
        if col_name not in hf:
            datashape = (None, )
            if (dataset.ndim > 1):
                datashape = (None, ) + dataset.shape[1:]
            hf.create_dataset(col_name, data=dataset, maxshape=datashape, compression=compression) # gzip, compression_opts=4
        else:
            hf[col_name].resize((hf[col_name].shape[0]) + dataset.shape[0], axis = 0)
            hf[col_name][-dataset.shape[0]:] = dataset