'''
    ab_hmdb51_splits.py

    Jason Corso

    Train and test an SVM on the HMDB51 data.
    Uses the 3 splits provided by the HMDB creators
    Will produce the result statistic that we reported in the paper.

    The processed HMDB51 data is available at
    http://www.cse.buffalo.edu/~jcorso/r/actionbank

    MAKE sure that ../code is in your PYTHONPATH, i.e., export PYTHONPATH=../code
    before running this script

    ---- Information on the splits
        There are totally 153 files in this folder,
        [action]_test_split[1-3].txt  corresponding to three splits reported in the paper.
        The format of each file is
        [video_name] [id]
        The video is included in the training set if id is 1
        The video is included in the testing set if id is 2
        The video is not included for training/testing if id is 0
        There should be 70 videos with id 1 , 30 videos with id 2 in each txt file.
    ----

    The following three videos are corrupt and we do not use them (as of 30 May 2012)
      pour/How_to_pour_beer_pour_u_nm_np1_fr_goo_0.avi
      pour/How_to_pour_beer__eh__pour_u_nm_np1_fr_goo_0.avi
      talk/jonhs_netfreemovies_holygrail_talk_h_nm_np1_fr_med_6.avi
'''

import argparse
import glob
import gzip
import numpy as np
import os
import os.path
import random as rnd
import scipy.io as sio
import multiprocessing as mp

def loadsplit(classes,path,splitnumber):

    trainfiles = []
    testfiles = []
    for ci,c in enumerate(classes):
        fp = open(os.path.join(path,"%s_test_split%d.txt"%(c,splitnumber)))
        L = fp.readlines()
        for l in L:
            (name,op) = l.strip().split()

            if op == '1':
                trainfiles.append([os.path.join(c,name),ci])
            elif op == '2':
                testfiles.append([os.path.join(c,name),ci])
        fp.close()

    return trainfiles,testfiles

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to perform 10-fold cross-validation on the HMDB51 data set using the included SVM code.",
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("root", help="path to the directory containing the action bank processed hmdb51 files structured as in root/class/class00_banked.npy.gz for each class")
    parser.add_argument("splits", help="path to the directory containing the HMDB51 splits files (153 of them)")

    args = parser.parse_args()

    vlen = 0

    classes = os.listdir(args.root)

    if (len(classes) != 51):
        print "error: found %d classes, but there should be 51"%(len(cdir))

    accs = np.zeros(3)

    for splitnumber in range(1,4):
        print "working on split %d"%splitnumber

        trainfiles,testfiles = loadsplit(classes,args.splits,splitnumber)

        print "have %d training files" % len(trainfiles)
        print "have %d testing files" % len(testfiles)

        if not vlen:
            fp = gzip.open(os.path.join(args.root,'%s%s'%(trainfiles[0][0],banked_suffix)),"rb")
            vlen = len(np.load(fp))
            fp.close()
            print "vector length is %d"%vlen

        Dtrain = np.zeros( (len(trainfiles),vlen), np.uint8 )
        Ytrain = np.ones ( (len(trainfiles)   )) * -1000

        for fi,f in enumerate(trainfiles):
            #print f
            fp = gzip.open(os.path.join(args.root,'%s%s'%(f[0],banked_suffix)),"rb")
            Dtrain[fi][:] = np.load(fp)
            fp.close()
            Ytrain[fi] = f[1]

        Dtest = np.zeros( (len(testfiles),vlen), np.uint8 )
        Ytest = np.ones ( (len(testfiles)   )) * -1000

        for fi,f in enumerate(testfiles):
            #print f
            fp = gzip.open(os.path.join(args.root,'%s%s'%(f[0],banked_suffix)),"rb")
            Dtest[fi][:] = np.load(fp)
            fp.close()
            Ytest[fi] = f[1]

        print Dtrain.shape
        print Ytrain.shape
        print Dtest.shape
        print Ytest.shape

        res=ab_svm.SVMLinear(Dtrain,np.int32(Ytrain),Dtest,threads=mp.cpu_count()-1,useLibLinear=True,useL1R=False)
        tp=np.sum(res==Ytest)
        print 'Accuracy is %.1f%%' % ((np.float64(tp)/Dtest.shape[0])*100)
        accs[splitnumber-1] = ((np.float64(tp)/Dtest.shape[0])*100)

    print 'Mean accuracy is %f'%(accs.mean())
