from __future__ import division
import os, glob
import numpy as np
import h5py
from skimage.transform import resize, warp
from transforms3d.euler import euler2mat
from transforms3d.affines import compose
import nibabel as nib

def get_random_transformation():
    T = [0, np.random.uniform(-8, 8), np.random.uniform(-8, 8)]
    R = euler2mat(np.random.uniform(-5, 5) / 180.0 * np.pi, 0, 0, 'sxyz')
    Z = [1, np.random.uniform(0.9, 1.1), np.random.uniform(0.9, 1.1)]
    A = compose(T, R, Z)
    return A

def get_tform_coords(im_size):
    coords0, coords1, coords2 = np.mgrid[:im_size[0], :im_size[1], :im_size[2]]
    coords = np.array([coords0 - im_size[0] / 2, coords1 - im_size[1] / 2, coords2 - im_size[2] / 2])
    return np.append(coords.reshape(3, -1), np.ones((1, np.prod(im_size))), axis=0)

def remove_low_high(im_input):
    im_output = im_input
    low = np.percentile(im_input, 1)
    high = np.percentile(im_output, 99)
    im_output[im_input < low] = low
    im_output[im_input > high] = high
    return im_output

def normalize(im_input):
    x_start = im_input.shape[0] // 4
    x_range = im_input.shape[0] // 2
    y_start = im_input.shape[1] // 4
    y_range = im_input.shape[1] // 2
    z_start = im_input.shape[2] // 4
    z_range = im_input.shape[2] // 2
    roi = im_input[x_start : x_start + x_range, y_start : y_start + y_range, z_start : z_start + z_range]
    im_output = (im_input - np.mean(roi)) / np.std(roi)
    return im_output

def read_seg(path):
    seg = nib.load(glob.glob(os.path.join(path, '*_LesionSmooth_stx.nii.gz'))[0]).get_data().astype(np.float32)
    return seg

def read_image(path, is_training=True):
    t1 = nib.load(glob.glob(os.path.join(path, '*_corrected.nii.gz'))[0]).get_data().astype(np.float32)

    if is_training:
        seg = nib.load(glob.glob(os.path.join(path, '*_LesionSmooth_stx.nii.gz'))[0]).get_data().astype(np.float32)
        assert t1.shape == seg.shape
        nchannel = 2
    else:
        nchannel = 1

    image = np.empty((t1.shape[0], t1.shape[1], t1.shape[2], nchannel), dtype=np.float32)

    #image[..., 0] = remove_low_high(t1)
    image[..., 0] = normalize(t1)

    if is_training:
        image[..., 1] = seg

    return image

def read_patch(path):
    image = np.load(path + '.npy')
    seg = image[..., -1]
    label = np.zeros((image.shape[0], image.shape[1], image.shape[2], 2), dtype=np.float32)
    label[seg == 0, 0] = 1
    label[seg > 0, 1] = 1 #changed
    return image[..., :-1], label

def generate_patch_locations(patches, patch_size, im_size):
    nx = round((patches * 8 * im_size[0] * im_size[0] / im_size[1] / im_size[2]) ** (1.0 / 3))
    ny = round(nx * im_size[1] / im_size[0])
    nz = round(nx * im_size[2] / im_size[0])
    x = np.rint(np.linspace(patch_size, im_size[0] - patch_size, num=nx))
    y = np.rint(np.linspace(patch_size, im_size[1] - patch_size, num=ny))
    z = np.rint(np.linspace(patch_size, im_size[2] - patch_size, num=nz))
    return x, y, z

def generate_test_locations(patch_size, stride, im_size):
    stride_size_x = patch_size[0] / stride
    stride_size_y = patch_size[1] / stride
    stride_size_z = patch_size[2] / stride
    pad_x = (int(patch_size[0] / 2), int(np.ceil(im_size[0] / stride_size_x) * stride_size_x - im_size[0] + patch_size[0] / 2))
    pad_y = (int(patch_size[1] / 2), int(np.ceil(im_size[1] / stride_size_y) * stride_size_y - im_size[1] + patch_size[1] / 2))
    pad_z = (int(patch_size[2] / 2), int(np.ceil(im_size[2] / stride_size_z) * stride_size_z - im_size[2] + patch_size[2] / 2))
    x = np.arange(patch_size[0] / 2, im_size[0] + pad_x[0] + pad_x[1] - patch_size[0] / 2 + 1, stride_size_x)
    y = np.arange(patch_size[1] / 2, im_size[1] + pad_y[0] + pad_y[1] - patch_size[1] / 2 + 1, stride_size_y)
    z = np.arange(patch_size[2] / 2, im_size[2] + pad_z[0] + pad_z[1] - patch_size[2] / 2 + 1, stride_size_z)
    return (x, y, z), (pad_x, pad_y, pad_z)

def perturb_patch_locations(patch_locations, radius):
    x, y, z = patch_locations
    x = np.rint(x + np.random.uniform(-radius, radius, len(x)))
    y = np.rint(y + np.random.uniform(-radius, radius, len(y)))
    z = np.rint(z + np.random.uniform(-radius, radius, len(z)))
    return x, y, z

def generate_patch_probs(path, patch_locations, patch_size, im_size):
    x, y, z = patch_locations
    seg = nib.load(glob.glob(os.path.join(path, '*_LesionSmooth_stx.nii.gz'))[0]).get_data().astype(np.float32)
    p = []
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                patch = seg[int(x[i] - patch_size / 2) : int(x[i] + patch_size / 2),
                            int(y[j] - patch_size / 2) : int(y[j] + patch_size / 2),
                            int(z[k] - patch_size / 2) : int(z[k] + patch_size / 2)]
                patch = (patch > 0).astype(np.float32)
                percent = np.sum(patch) / (patch_size * patch_size * patch_size)
                p.append((1 - np.abs(percent - 0.5)) * percent)
    p = np.asarray(p, dtype=np.float32)
    p[p == 0] = np.amin(p[np.nonzero(p)])
    p = p / np.sum(p)
    return p
