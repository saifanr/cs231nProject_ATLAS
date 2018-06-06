import os, sys
import numpy as np
from utils import read_image

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception("Need at least the input directory")
    input_dir = sys.argv[1]

    all_paths = []
    for dirpath, dirnames, files in os.walk(input_dir):
        if os.path.basename(dirpath)[0:3] == 't01':
            all_paths.append(dirpath)

    count = 0
    for path in all_paths:
        count+=1
        probs = np.load(os.path.join(path, 'probs.npy'))
        pred = np.argmax(probs, 3).astype(np.float32)
        print('Saving image no.', count)
        np.save(os.path.join(path, 'preds'), pred)
