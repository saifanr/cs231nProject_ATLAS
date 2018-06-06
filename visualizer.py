import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import os, sys
from utils import read_image

def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

def multi_slice_viewer(volume):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[1] // 2
    ax.imshow(volume[:,ax.index,:], cmap='gray')
    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.show()

def full_multi_viewer(img, seg_mask, prediction):
    remove_keymap_conflicts({'j', 'k'})
    fig, (ax_img, ax_seg, ax_pre) = plt.subplots(1,3)
    ax_img.volume = img
    ax_seg.volume = seg_mask
    ax_pre.volume = prediction
    ax_img.index = ax_img.volume.shape[1] // 2
    ax_seg.index = ax_seg.volume.shape[1] // 2
    ax_pre.index = ax_pre.volume.shape[1] // 2
    ax_img.imshow(ax_img.volume[:,ax_img.index,:], cmap='gray')
    ax_seg.imshow(ax_seg.volume[:,ax_seg.index,:], cmap='gray', norm=clr.NoNorm())
    ax_pre.imshow(ax_pre.volume[:,ax_pre.index,:], cmap='gray', norm=clr.NoNorm())
    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.show()

def patch_viewer(patch):
    remove_keymap_conflicts({'j', 'k'})
    fig, (ax_img, ax_seg) = plt.subplots(1,2)
    ax_img.volume = patch[...,0]
    ax_seg.volume = patch[...,1]
    ax_img.index = ax_img.volume.shape[1] // 2
    ax_seg.index = ax_seg.volume.shape[1] // 2
    ax_img.imshow(ax_img.volume[:,ax_img.index,:], cmap='gray')
    ax_seg.imshow(ax_seg.volume[:,ax_seg.index,:], cmap='gray')
    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.show()

def process_key(event):
    fig = event.canvas.figure
    for ax in fig.axes:
        if event.key == 'j':
            previous_slice(ax)
        elif event.key == 'k':
            next_slice(ax)
        fig.canvas.draw()
    plt.show()

def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[1]  # wrap around using %
    ax.images[0].set_array(volume[:,ax.index,:])

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[1]
    ax.images[0].set_array(volume[:,ax.index,:])

if __name__ == '__main__':

    if len(sys.argv) < 2:
        raise Exception("Need the patch directory")

    # patch_path = sys.argv[1]

    # patch = np.load( patch_path )
    image_path = sys.argv[1]
    pred_path = sys.argv[2]

    image = read_image( image_path )

    img = image[...,0]
    seg = image[...,1]

    pred = np.load( pred_path ).astype(np.float32)

    pred[pred>0.5] = 0.9
    seg[seg>0.5] = 0.9

    full_multi_viewer(img, seg, pred)

    # else:
    #    img = np.load( FLAGS.patch_path )[:,:,:,1]

    # patch_viewer( patch )
