from sklearn.multiclass import OneVsRestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn import preprocessing
import multiprocessing as multi
from sklearn.svm import SVC
import os.path as path
import numpy as np
import sys
import time
import os
import glob
import gzip

def load_simpleone(root, suffix="_banked.npy.gz"):
    """
    Code to load banked vectors at top-level directory root into a feature matrix and class-label vector.

    Classes are assumed to each exist in a single directory just under root.
    Example: root/jump, root/walk would have two classes "jump" and "walk" and in each
    root/X directory, there are a set of _banked.npy.gz files created by the actionbank.py
    script.

    For other more complex data set arrangements, you'd have to write some custom code, this is
    just an example.

    A feature matrix D and label vector Y are returned.  Rows and D and Y correspond.
    You can use scipy.io to save these as .mat files if you want to export to matlab...
    """

    classdirs = os.listdir(root)
    classdirs.sort()
    vlen=0 # length of each bank vector, we'll get it by loading one in...
    Ds = []
    Ys = []

    for ci,c in enumerate(classdirs):
        cd = os.path.join(root,c)
        files = glob.glob(os.path.join(cd,'*%s'%suffix))
        files.sort()
        print "%d files in %s" %(len(files),cd)

        if not vlen:
            fp = gzip.open(files[0],"rb")
            filedata = np.load(fp)
            vlen = len(filedata)
            filetype = type(filedata[0])
            fp.close()
            print "vector length is %d" % (vlen)

 #       Di = np.zeros( (len(files),vlen), np.uint8)
        Di = np.zeros( (len(files),vlen), filetype)
        Yi = np.ones ( (len(files)   ), np.uint8) * ci

        for bi,b in enumerate(files):
            fp = gzip.open(b,"rb")
            Di[bi][:] = np.load(fp)
            fp.close()

        Ds.append(Di)
        Ys.append(Yi)

    D = Ds[0]
    Y = Ys[0]
    for i,Di in enumerate(Ds[1:]):
        D = np.vstack( (D,Di) )
        Y = np.concatenate( (Y,Ys[i+1]) )

    return D,Y

def load_groups(groups, training, testing, scale=False, verbose=False):
    """
    Input:
    - full group sets

    Returns train & test feature and label matrices
    """
    print "Training", training
    print "Testing", testing

    fp = gzip.open(groups[0][0][0])
    filedata = np.load(fp)
    vlen = len(filedata)
    filetype = type(filedata[0])
    if verbose:
        print filetype
        print "Sample data before scaling"
        print filedata
    fp.close()
    print "vector length is %d"%vlen

    nt = 0
    for i in training:
        nt += len(groups[i])
    print "have %d training files"%nt
    Dtrain = np.zeros( (nt,vlen), filetype )
    Ytrain = np.ones ( (nt), np.int8 ) * -1000

    ti = 0
    for i in training:
        for j in groups[i]:
            # j[0] is path and j[1] is class
            fp = gzip.open(j[0])
            Dtrain[ti][:] = np.load(fp)
            fp.close()
            Ytrain[ti] = j[1]
            ti += 1
#            print j

    ne = 0
    for i in testing:
        ne += len(groups[i])
    print "have %d testing files"%ne
    Dtest = np.zeros( (ne,vlen), filetype )
    Ytest = np.ones ( (ne), np.int8 ) * -1000

    ti = 0
    for i in testing:
        for j in groups[i]:
            # j[0] is path and j[1] is class
            fp = gzip.open(j[0])
            Dtest[ti][:] = np.load(fp)
            fp.close()
            Ytest[ti] = j[1]
            ti += 1

    if scale:
        min_max_scaler = preprocessing.MinMaxScaler(copy=False)
        Dtrain = min_max_scaler.fit_transform(Dtrain.astype(float))
        Dtest = min_max_scaler.transform(Dtest.astype(float))

    if verbose:
        print "Train data scaled"
        print Dtrain
        print "Test data scaled"
        print Dtest
        sys.stdout.flush()
    return Dtrain, Dtest, Ytrain, Ytest

# main group-wise testing routine
def testgroups(groups,training,testing, scale=True, verbose=False, useShogun=False):
    """
    Comment
    """

    print "Training", training
    print "Testing", testing

    fp = gzip.open(groups[0][0][0])
    filedata = np.load(fp)
    vlen = len(filedata)
    filetype = type(filedata[0])
    if verbose:
        print filetype
        print "Sample data before scaling"
        print filedata
    fp.close()
    print "vector length is %d"%vlen

    nt = 0
    for i in training:
        nt += len(groups[i])
    print "have %d training files"%nt
    Dtrain = np.zeros( (nt,vlen), filetype )
    Ytrain = np.ones ( (nt) ) * -1000

    ti = 0
    for i in training:
        for j in groups[i]:
            # j[0] is path and j[1] is class
            fp = gzip.open(j[0])
            Dtrain[ti][:] = np.load(fp)
            fp.close()
            Ytrain[ti] = j[1]
            ti += 1

    ne = 0
    for i in testing:
        ne += len(groups[i])
    print "have %d testing files"%ne
    Dtest = np.zeros( (ne,vlen), filetype )
    Ytest = np.ones ( (ne) ) * -1000

    ti = 0
    for i in testing:
        for j in groups[i]:
            # j[0] is path and j[1] is class
            fp = gzip.open(j[0])
            Dtest[ti][:] = np.load(fp)
            fp.close()
            Ytest[ti] = j[1]
            ti += 1

    if scale:
        min_max_scaler = preprocessing.MinMaxScaler(copy=False)
        Dtrain = min_max_scaler.fit_transform(Dtrain.astype(float))
        Dtest = min_max_scaler.transform(Dtest.astype(float))
        if verbose:
            print "Train data scaled"
            print Dtrain
            print "Test data scaled"
            print Dtest
            sys.stdout.flush()

    if useShogun:
        res=ab_svm.SVMLinear(Dtrain,np.int32(Ytrain),Dtest,threads=multi.cpu_count()-1,useLibLinear=True,useL1R=False)
    else:
        clf = OneVsRestClassifier(LinearSVC(C=5e3), -1).fit(Dtrain, np.int32(Ytrain))
        res = clf.predict(Dtest)

    tp=np.sum(res==Ytest)
    print 'Accuracy is %.1f%%' % ((np.float64(tp)/Dtest.shape[0])*100)
    return ((np.float64(tp)/Dtest.shape[0])*100), res

def test_ucf50_5fold(datapath_root, C=5e3, gamma=None):
    """
    Loads all 5 sets for the feature (action bank or object bank), trains and test
    Returns: mean accuracy
    """
    accs = np.zeros(5)
    tts = np.zeros(5)

    for i in range(5):
        datafiles = np.load(os.path.join(datapath_root, 'set%d.npz' % i))
        Dtrain = datafiles['Dtrain']
        Ytrain = datafiles['Ytrain']
        Dtest = datafiles['Dtest']
        Ytest = datafiles['Ytest']

        vlen = Dtrain.shape[1]
        print "vector length is %d"%vlen
        t0 = time.time()
        if gamma is None:
            clf = OneVsRestClassifier(LinearSVC(C=C), -1).fit(Dtrain, np.int32(Ytrain))
        else:
            clf = OneVsRestClassifier(SVC(kernel='rbf', cache_size=2000, C=C, gamma=gamma), -1).fit(Dtrain, np.int32(Ytrain))
        pred = clf.predict(Dtest)

        tp=np.sum(pred==Ytest)
        print 'Accuracy is %.1f%%' % ((np.float64(tp)/Dtest.shape[0])*100)
        accs[i] = ((np.float64(tp)/Dtest.shape[0])*100)
        tts[i] = time.time() - t0

    return accs.mean(), tts.mean()

def param_gridsearch(dataroot, nfolds=3):
    """
    Perform grid search on full data set for parameter optimization
    """
    data = np.load(os.path.join(dataroot, 'full.npz'))
    Dtrain = data['Dtrain']
    Ytrain = data['Ytrain']
    print("Fitting the classifier to the training set")
    '''
    param_grid = [
      {'estimator__C': [1, 10, 1e2, 1e3, 1e4, 1e5, 5e3], 'estimator__kernel': ['linear']},
      {'estimator__C': [1, 10, 1e2, 1e3, 1e4, 1e5, 5e3], 'estimator__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], 'estimator__kernel': ['rbf']},
     ]
    '''
    param_grid = {'estimator__C': [1, 10, 100, 1e3, 5e3, 1e4, 5e4, 1e5],}
    clf = GridSearchCV(OneVsRestClassifier(SVC(kernel='linear', cache_size=2000, class_weight=None, probability=True), -1), param_grid, n_jobs=-1, cv=nfolds)
    clf = clf.fit(Dtrain, Ytrain)
    print clf.best_estimator_
    print clf.best_params_
    print clf.best_score_
    return clf

def savegroups(groups,training,testing, setnumber, scale=True, verbose=False, outfolder="ab_npy_sets"):
    """
    Saves all individual ucf50 features and saves them into proper compressed sets for 5-fold cv
    """
    print "Training", training
    print "Testing", testing

    fp = gzip.open(groups[0][0][0])
    filedata = np.load(fp)
    vlen = len(filedata)
    filetype = type(filedata[0])
    if verbose:
        print filetype
        print "Sample data before scaling"
        print filedata
    fp.close()
    print "vector length is %d"%vlen

    nt = 0
    for i in training:
        nt += len(groups[i])
    print "have %d training files"%nt
    Dtrain = np.zeros( (nt,vlen), filetype )
    Ytrain = np.ones ( (nt), np.int8) * -1000

    ti = 0
    for i in training:
        for j in groups[i]:
            # j[0] is path and j[1] is class
            fp = gzip.open(j[0])
            Dtrain[ti][:] = np.load(fp)
            fp.close()
            Ytrain[ti] = j[1]
            ti += 1

    ne = 0
    for i in testing:
        ne += len(groups[i])
    print "have %d testing files"%ne
    Dtest = np.zeros( (ne,vlen), filetype )
    Ytest = np.ones ( (ne), np.int8 ) * -1000

    ti = 0
    for i in testing:
        for j in groups[i]:
            # j[0] is path and j[1] is class
            fp = gzip.open(j[0])
            Dtest[ti][:] = np.load(fp)
            fp.close()
            Ytest[ti] = j[1]
            ti += 1

    if scale:
        min_max_scaler = preprocessing.MinMaxScaler(copy=False)
        Dtrain = min_max_scaler.fit_transform(Dtrain.astype(float))
        Dtest = min_max_scaler.transform(Dtest.astype(float))
        if verbose:
            print "Train data scaled"
            print Dtrain
            print "Test data scaled"
            print Dtest
            sys.stdout.flush()

    np.savez("%s/set%d" % (outfolder, setnumber), Dtrain=Dtrain, Dtest=Dtest, Ytrain=Ytrain, Ytest=Ytest, training=training, testing=testing)

def load_save_groups(root, suffix, useLooCV=False, scale=True, outfolder='ob_pooled_npy_sets', verbose=False):
    """
    Loads all individual ucf50 features and saves them into proper compressed sets for 5-fold cv
    """
    cdir = os.listdir(root)

    if (len(cdir) != 50):
        print "error: found %d classes, but there should be 50"%(len(cdir))

    groups = []
    for g in range(1,26):
        gset = []
        for ci,cl in enumerate(cdir):
            files = glob.glob(os.path.join(root,cl,'*g%02d*%s'%(g,suffix)))
            for f in files:
                gset.append( [f,ci] )
        print "group %d has %d"%(g,len(gset))
        groups.append(gset)

    full = np.arange(25)

    sets = []

    if useLooCV:
        for i in np.arange(25):
            sets.append([i])
    else:
        sets.append(np.arange(0,5))
        sets.append(np.arange(5,10))
        sets.append(np.arange(10,15))
        sets.append(np.arange(15,20))
        sets.append(np.arange(20,25))
        for setnumber in range(5):
            training = np.setdiff1d(full, sets[setnumber])
            testing = sets[setnumber]
            print "Training", training
            print "Testing", testing

            fp = gzip.open(groups[0][0][0])
            filedata = np.load(fp)
            vlen = len(filedata)
            filetype = type(filedata[0])
            if verbose:
                print filetype
                print "Sample data before scaling"
                print filedata
            fp.close()
            print "vector length is %d"%vlen

            nt = 0
            for i in training:
                nt += len(groups[i])
            print "have %d training files"%nt
            Dtrain = np.zeros( (nt,vlen), filetype )
            Ytrain = np.ones ( (nt), np.int8) * -1000

            ti = 0
            for i in training:
                for j in groups[i]:
                    # j[0] is path and j[1] is class
                    fp = gzip.open(j[0])
                    Dtrain[ti][:] = np.load(fp)
                    fp.close()
                    Ytrain[ti] = j[1]
                    ti += 1

            ne = 0
            for i in testing:
                ne += len(groups[i])
            print "have %d testing files"%ne
            Dtest = np.zeros( (ne,vlen), filetype )
            Ytest = np.ones ( (ne), np.int8 ) * -1000

            ti = 0
            for i in testing:
                for j in groups[i]:
                    # j[0] is path and j[1] is class
                    fp = gzip.open(j[0])
                    Dtest[ti][:] = np.load(fp)
                    fp.close()
                    Ytest[ti] = j[1]
                    ti += 1

            if scale:
                min_max_scaler = preprocessing.MinMaxScaler(copy=False)
                Dtrain = min_max_scaler.fit_transform(Dtrain.astype(float))
                Dtest = min_max_scaler.transform(Dtest.astype(float))
                if verbose:
                    print "Train data scaled"
                    print Dtrain
                    print "Test data scaled"
                    print Dtest
                    sys.stdout.flush()

            if not os.path.exists(outfolder):
                os.makedirs(outfolder)
            np.savez("%s/set%d" % (outfolder, setnumber), Dtrain=Dtrain, Dtest=Dtest, Ytrain=Ytrain, Ytest=Ytest, training=training, testing=testing)

def split_data(root, suffix='_banked.npy.gz', useLooCV=False):
    """
    Gets 25 groups from the ucf50 dataset

    Inputs:
    - root: String for root path to the .npy saved feature files
    - suffix: extension of files to load

    Returns:
    - groups: K x N. K groups with each N videos
    - full: 1-D array of length K. [1..25]
    - sets: Folds to test on
    """
    cdir = os.listdir(root)

    if (len(cdir) != 50):
        print "error: found %d classes, but there should be 50"%(len(cdir))

    groups = []
    for g in range(1,26):
        gset = []
        for ci,cl in enumerate(cdir):
            files = glob.glob(os.path.join(root,cl,'*g%02d*%s'%(g,suffix)))
            for f in files:
                gset.append( [f,ci] )
#        print "group %d has %d"%(g,len(gset))
        groups.append(gset)

    full = np.arange(25)
    sets = []

    if useLooCV:
        for i in np.arange(25):
            sets.append([i])
    else:
        sets.append(np.arange(0,5))
        sets.append(np.arange(5,10))
        sets.append(np.arange(10,15))
        sets.append(np.arange(15,20))
        sets.append(np.arange(20,25))

    return groups, full, sets

def save_fv(fv_root, output="fv_ucf50_python/"):
    """
    Loads each fv descriptor and concatenates into a matrix with feature
    vectors for each video. Saves it in compress numpy format npy.gz

    Inputs:
    - root: String for root path of extracted fisher vector (compute_fv output)
            Files in root have the *descriptor.fv.txt extension.

    """
    classdirs = os.listdir(fv_root)
    classdirs.sort()
    vlen = 0

    for ci, c in enumerate(classdirs):
        cd = os.path.join(fv_root, c)
        hof_files = glob.glob(os.path.join(cd,'*hof.fv.txt'))
        hog_files = glob.glob(os.path.join(cd,'*hog.fv.txt'))
        mbh_files = glob.glob(os.path.join(cd,'*mbhx.fv.txt'))
        tr_files  = glob.glob(os.path.join(cd,'*traj.fv.txt'))

        if not vlen:
            X_hof = np.loadtxt(hof_files[0])
            X_hog = np.loadtxt(hog_files[0])
            X_mbh = np.loadtxt(mbh_files[0])
            X_tr  = np.loadtxt(tr_files[0])
            vlen = X_hof.shape[0] + X_hog.shape[0] + X_mbh.shape[0] + X_tr.shape[0]
            print 'hof: %d | hog: %d | mbh %d | tr %d' % (X_hof.shape[0], X_hog.shape[0],
                                                          X_mbh.shape[0], X_tr.shape[0])
            print 'Has feature vector of size %d' % vlen

        Di = np.zeros((len(hof_files), vlen), np.float32)
        Yi = np.ones((len(hof_files)), np.uint8) * ci
        hof_files.sort(), hog_files.sort(), mbh_files.sort(), tr_files.sort()
        for i, (hof, hog, mbh, tr) in enumerate(zip(hof_files, hog_files,
                                                    mbh_files, tr_files)):
            X_hof = np.loadtxt(hof)
            X_hog = np.loadtxt(hog)
            X_mbh = np.loadtxt(mbh)
            X_tr  = np.loadtxt(tr)
            Di[i] = np.hstack((X_hof, X_hog, X_mbh, X_tr))

            # power-normalization
            Di[i] = np.sign(Di[i]) * np.abs(Di[i]) ** 0.5

            # L2 normalize
            norms = np.sqrt(np.sum(Di[i] ** 2))
            Di[i] /= np.ones(len(Di[i])) * norms
            # handle images with 0 local descriptor (100 = far away from "normal" images)
            Di[i][np.isnan(Di[i])] = 100

            fname = os.path.basename(hof)[:-11]
            if not os.path.exists(os.path.join(output, c)):
              os.makedirs(os.path.join(output, c))

            outfile = os.path.join(output, c, fname + '_fv.npy.gz')
            of = gzip.open(outfile, "wb")
            np.save(of, Di[i])
            of.close()
