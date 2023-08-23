import SimpleITK as sitk

def read_mhd(filename):
    '''
        https://stackoverflow.com/questions/37290631/reading-mhd-raw-format-in-python
        This function reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image.
    '''
    itkimage = sitk.ReadImage(filename)
    img = sitk.GetArrayFromImage(itkimage)
    return img

def write_mhd(img, filepath):
    # TODO: check if img is np.uint8
    sitk.WriteImage(sitk.GetImageFromArray(img), filepath)