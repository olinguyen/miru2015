# extract DT features on the fly and select feature points randomly
# need select_pts and dt binaries

import subprocess, os, ffmpeg
import sys
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import datetime
import fnmatch
import os

# Dense Trajectories binary
dtBin = '/home/syq/fudan/NTT_report_codes/codes/DenseTraj/dtfv/script/DenseTrackStab'
# Compiled fisher vector binary
fvBin = '/home/syq/fudan/NTT_report_codes/codes/DenseTraj/dtfv/script/compute_fv'
# Dense Trajectories extract binary
dtextractBin = '/home/syq/fudan/trajMF_code/extract'
# TrajMF binary
trajmfBin = '/home/syq/fudan/trajMF_code/mf'

# Temp directory to store resized videos
tmpDir = '/mnt/share04/oliver/tmp/'
# PCA list
pcaList = '/home/syq/fudan/NTT_report_codes/codes/DenseTraj/medcodebook/med.pca.lst'
# GMM list
codeBookList = '/home/syq/fudan/NTT_report_codes/codes/DenseTraj/medcodebook/data/med.codebook.lst'
# outputBase
#outputPath = '/mnt/share04/oliver/ucf50-dt/'
outputPath = '/home/syq/research_final/data/dense-traj/fv'
dt_suffix = '.gz'
# Bounding box path
bbPath = '/home/watanabe.yukito/olivier/NTT_report_codes/codes/DenseTraj/dtfv/script/bb_file/HMDB51/'
# Path to input videos
#videopath = "/mnt/share04/oliver/videos/ucf50/"
videopath = "/mnt/share04/oliver/videos/hmdb51/"
dt_path = '/home/syq/research_final/data/dense-traj/ucf50-dt/'

def extract(videoName):
    '''
        Extracts the IDTFs and stores them in outputBase file.
    '''
    category = os.path.basename(os.path.dirname(videoName))
    outputBase = os.path.join(outputPath, category, os.path.basename(videoName[:-4] + dt_suffix))
    if not os.path.exists(os.path.join(outputPath, category)):
        os.makedirs(os.path.join(outputPath, category))
    if not os.path.exists(videoName):
        print '%s does not exist!' % videoName
        return False
    resizedName = videoName
    bbFile = os.path.join(bbPath, os.path.basename(videoName)[:-4] + '.bb')
    cmd = '%s "%s" -H "%s" | gzip > "%s"' % (dtBin, resizedName, bbFile, outputBase)
#    print cmd
    subprocess.call(cmd, shell=True)
#    subprocess.call('%s %s -H %s | gzip > %s' % (dtBin, resizedName, bbFile, outputBase), shell=True)
#   subprocess.call('%s %s -H %s > %s' % (dtBin, resizedName, bbFile, outputBase), shell=True)
#    print '%s done..' % outputBase
    return True

def extract_fv_from_dt(densetrajfile):
    '''
        Computes the fisher vector from the raw dense trajectories
    '''
    category = os.path.basename(os.path.dirname(densetrajfile))
    outputPath = '/home/syq/research_final/data/dense-traj/fv'
    outputBase = os.path.join(outputPath, category, os.path.basename(densetrajfile[:-4]))
    if not os.path.exists(os.path.join(outputPath, category)):
        os.makedirs(os.path.join(outputPath, category))
    if not os.path.exists(densetrajfile):
        print '%s does not exist!' % densetrajfile
        return False
    cmd = 'gunzip "%s" | "%s" "%s" "%s" "%s"' % (densetrajfile, fvBin, pcaList, codeBookList, outputBase)
    print cmd
    '''
    subprocess.call(cmd, shell=False)
    return True
    '''

def extract_trajmf(densetrajfile):
    '''
        Computes the TrajMF features from the raw dense trajectories
    '''
    featureTypes = ['tr', 'hog', 'hof', 'mbh']

    dt_FeatureFileName = os.path.basename(densetrajfile)

    dt_PartFeaturePath = '/home/syq/research_final/data/dense-traj/ucf50_mf/part-features'
    mfpath = '/home/syq/research_final/data/dense-traj/ucf50_mf/mf'

    cbroot = os.path.dirname(trajmfBin)

    # Create directories for dense traj part features and MF output
    for feature in featureTypes:
        partFeatOutpath = os.path.join(dt_PartFeaturePath, feature)
        mfFeatOutpath = os.path.join(mfpath, feature)
        if not os.path.exists(partFeatOutpath):
            os.makedirs(partFeatOutpath)
        if not os.path.exists(mfFeatOutpath):
            os.makedirs(mfFeatOutpath)

    # part the Dense Trajectories feature into 4 parts (TrajShape, HOG, HOF, MBH)
    extract_cmd = '%s %s %s %s' % (dtextractBin, densetrajfile, dt_PartFeaturePath, dt_FeatureFileName)
    subprocess.call(extract_cmd, shell=True)

    # get TrajMF for hog, hof and mbh
    for feature in featureTypes[1:]:
        inPath = os.path.join(dt_PartFeaturePath, feature) # path for the input feature file, e.g. dt_feature/hog
        mfFeatOutpath = os.path.join(mfpath, feature)
        cbFile = '%s300Centers.txt' % feature

        if feature == 'hog':
            nDim = 96
        elif feature == 'hof':
            nDim = 108
        elif feature == 'mbh':
            nDim = 192
        else:
            print 'N/A feature type'

        mf_cmd = '%s %s %s %s %s %d' % (trajmfBin, dt_FeatureFileName, inPath, mfFeatOutpath, os.path.join(cbroot, cbFile), nDim)
        subprocess.call(mf_cmd, shell=True)
    return True

def extract_fv(videoName):
    outputBase = os.path.join(outputPath, os.path.basename(videoName))
    if not os.path.exists(videoName):
        print '%s does not exist!' % videoName
        return False
    if check_dup(outputBase):
        print '%s processed' % videoName
        return True
    resizedName = os.path.join(tmpDir, os.path.basename(videoName))
    if not ffmpeg.resize(videoName, resizedName):
        resizedName = videoName     # resize failed, just use the input video
    print dtBin, resizedName, fvBin, pcaList, codeBookList, outputBase
    starttime = datetime.datetime.now()
    subprocess.call('%s %s | %s %s %s %s' % (dtBin, resizedName, fvBin, pcaList, codeBookList, outputBase), shell=True)
    endtime = datetime.datetime.now()
    interval = (endtime-starttime).seconds
    print interval
    return True

def check_dup(outputBase):
    """
    Check if fv of all modalities have been extracted
    """
    featTypes = ['traj', 'hog', 'hof', 'mbhx', 'mbhy']
    featDims = [20, 48, 54, 48, 48]
    for i in range(len(featTypes)):
        featName = '%s.%s.fv.txt' % (outputBase, featTypes[i])
        if not os.path.isfile(featName) or not os.path.getsize(featName) > 0:
            return False
        # check if the length of feature can be fully divided by featDims
        f = open(featName)
        featLen = len(f.readline().rstrip().split())
        f.close()
        if featLen % (featDims[i] * 512) > 0:
            return False
    return True

def get_filelist(rootpath, extension='avi'):
    files = []
    for root, dirnames, filenames in os.walk(rootpath):
      for filename in fnmatch.filter(filenames, '*%s' % extension):
        files.append(os.path.join(root, filename))
    return files

if __name__ == '__main__':
    videoList = get_filelist(dt_path, '')
    try:
        videos = videoList
        pool = ThreadPool(20)
        pool.map(extract_trajmf, videos)
        pool.close()
        pool.join()
    except IOError:
        sys.exit(0)
