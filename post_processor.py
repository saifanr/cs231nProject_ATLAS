from __future__ import division
import os, sys
import numpy as np
import glob
import h5py
from skimage.transform import resize, warp
from transforms3d.euler import euler2mat
from transforms3d.affines import compose
import nibabel as nib

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception("Need at least the input directory")
    input_dir = sys.argv[1]

    all_paths = []
    for dirpath, dirnames, files in os.walk(input_dir):
        if os.path.basename(dirpath)[0:3] == 't01':
            all_paths.append(dirpath)

    for path in all_paths:

        preds = np.load(os.path.join(path, 'preds.npy'))
        mask = nib.load(os.path.join(path, 'mask_mask.nii.gz')).get_data().astype(np.float32)

        preds[ mask == 0 ] = 0

        print( str(path) )
        np.save(os.path.join(path, 'preds_processed'), preds)
