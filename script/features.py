import numpy as np
import gzip
import os

def feature_pool(dirpath, outdir, compute_mean=False, save=True, txt_flag=False):
  """
  Loop through input directory and compute max/mean pooling of feature vectors
  """
  for root, dirs, files in os.walk(dirpath, topdown = False):
    for file in files:
        x = np.loadtxt(path.join(root, file))
        # Hacky method of getting max because numpy returns garbage when max is called on 1row matrix
        if x.ndim > 1:
            if compute_mean == False:
                featPooled = x.max(0)
                outfilename = file.replace('.feat', '-max.feat')
            else:
                featPooled = x.mean(0)
                outfilename = file.replace('.feat', '-mean.feat')
        else:
            featPooled = x
        if save == True:
          outpath = os.path.join(root.replace(dirpath, outdir), outfilename)
          if txt_flag == True:
            featPooled.tofile(outpath, sep=' ')
          else:
            outpath += ".npy.gz"
            of = gzip.open(outpath, "wb")
            np.save(of, featPooled)
            of.close()
          print outpath

def chunks(l, n):
    """ Yield successive n-sized chunks from l."""
    return tuple(l[i:i+n] for i in xrange(0, len(l), n))

def pool_objectbank(feature_path, maxPool=False, isActionBank=False):
    if isActionBank == True:
        attributes_per_detector = 73
    else:
        attributes_per_detector = 252
    currfeat = np.load(gzip.open(feature_path, 'rb'))
    nchunks = chunks(currfeat, attributes_per_detector)
    pooled = []
    for chunk in nchunks:
        if maxPool == True:
            pooled.append(chunk.max())
        else:
            pooled.append(chunk.mean())
    return pooled

def pool_ob_videoclip(feature_path):
    """ Pool txt feature file """
    num_detectors = 177
    attributes_per_detector = 252
    videofeat = np.loadtxt(feature_path)
    # pool all object detectors for each image in the clip
    pooled_video = np.empty((0,num_detectors), np.float32)
    for imagefeat in videofeat:
        nchunks = chunks(imagefeat, attributes_per_detector)
        pooled_image = []
        for chunk in nchunks:
            pooled_image.append(chunk.mean())
        pooled_video = np.vstack((pooled_video, pooled_image))

    # Max-pool all frames from the video to represent in a single feature vector
    max_pooled = pooled_video.max(0)
    return max_pooled

def main():
    """ Pool object bank features """
    ob_feats_path = '/home/syq/research_final/data/features/ob_hmdb51_python'
    max_pooled_path = '/home/syq/research_final/data/features/ob_hmdb51_pooled_python'
    for root, dirs, files in os.walk(ob_feats_path):
        for file in files:
            outpath = root.replace(ob_feats_path, max_pooled_path)
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            featpath = path.join(root, file)
            objectbank_pooled = pool_objectbank(featpath, maxPool=False, isActionBank=False)
            print featpath
            print objectbank_pooled
            raw_input()
            outfile = gzip.open(path.join(outpath, file), 'wb')
            np.save(outfile, objectbank_pooled)
            outfile.close()

    """ Pool action bank features """
    ab_feats_path = '/home/syq/research_final/data/features/ab_hmdb51_e1f1g2/'
    mean_pooled_path = '/home/syq/research_final/data/features/ab_hmdb51_pooled_python/'
    for root, dirs, files in os.walk(ab_feats_path):
        for file in files:
            outpath = root.replace(ab_feats_path, mean_pooled_path)
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            featpath = path.join(root, file)
            actionbank_pooled = pool_objectbank(featpath, maxPool=False, isActionBank=True)
            outfile = gzip.open(path.join(outpath, file), 'wb')
            np.save(outfile, actionbank_pooled)
            outfile.close()

    """
    Mean pool each individual frame, then perform max-pooling to represent the video with a single vector
    """

    #ob_feats_path = '/home/syq/research_final/data/features/ob_ucf50_txt_feats'
    mean_pooled_path = '/home/syq/research_final/data/features/ob_hmdb51_pooled_python_v2'
    for root, dirs, files in os.walk(ob_feats_path):
        for file in files:
            outpath = root.replace(ob_feats_path, mean_pooled_path)
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            featpath = path.join(root, file)
            objectbank_pooled = pool_ob_videoclip(featpath)
            outfile = gzip.open(path.join(outpath, file+'.npy.gz'), 'wb')
            np.save(outfile, objectbank_pooled)
            outfile.close()
