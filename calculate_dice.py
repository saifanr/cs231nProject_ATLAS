import os, sys
import numpy as np
from utils import read_image

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception("Need at least the input directory")
    input_dir = sys.argv[1]

    f = open(os.path.join( input_dir, 'dice_scores.txt' ), 'w')

    all_paths = []
    for dirpath, dirnames, files in os.walk(input_dir):
        if os.path.basename(dirpath)[0:3] == 't01':
            all_paths.append(dirpath)

    dices = []

    for path in all_paths:
        pred = np.load(os.path.join(path, 'preds.npy'))
        truth = read_image( path )[...,1]

        pred[pred>0.5] = 1
        truth[truth>0.5] = 1

        if pred.shape != truth.shape:
            raise ValueError("Shape mismatch: pred and truth must have the same shape.")

        # Compute Dice coefficient
        intersection = np.logical_and(im1, im2)

        dice = 2.0 * intersection.sum() / (im1.sum() + im2.sum())

        dices.append( dice )

        print( str(path) + '\t' +  str(dice) + '\n' )
        f.write( str(path) + '\t' +  str(dice) + '\n' )

    print( 'The overall dice score is ' + '\t' + str( np.asarray(dices).mean() + '\n' ) )
    f.write( 'The overall dice score is ' + '\t' + str( np.asarray(dices).mean() + '\n' ) )
    f.close()
