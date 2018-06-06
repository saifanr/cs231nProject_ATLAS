import os, sys, glob

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
        os.system( 'bet '+str(glob.glob(os.path.join(path, '*_corrected.nii.gz'))[0])+' '+str(os.path.join(path, 'mask.nii.gz')) + ' -m' )
        print( 'bet '+str(glob.glob(os.path.join(path, '*_corrected.nii.gz'))[0])+' '+str(os.path.join(path, 'mask.nii.gz')) + ' -m' )
        count = count + 1
        print( 'Done:'+str(count) )
