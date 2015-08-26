from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.svm import SVC
import numpy as np
import utility
import time
import gzip
import os
import hmdb51_splits

def evaluate_ucf50():
    """ Run 5-fold cross-validation on the UCF50 dataset """
    fv_features = 'fv_ucf50_python/'
    accs = []
    groups, full, sets = utility.split_data(fv_features, suffix='_fv.npy.gz',
                                            useLooCV=False)
    for i in xrange(5):
        ts = time.time()
        features_train, features_test, labels_train, labels_test = \
                                        utility.load_groups(
                                        groups,np.setdiff1d(full,sets[i]),
                                        sets[i], scale=False, verbose=False)

        clf = OneVsRestClassifier(estimator=LinearSVC(C=100), n_jobs=6)
        clf.fit(features_train, labels_train)
        acc = clf.score(features_test, labels_test)
        print "Fold %d accuracy: %.3f" % (i, acc)
        print "Train & testing time %.3f" % (time.time() - ts)
        accs.append(acc)

    with open('fv_ucf50_accs_5fold.txt', 'w') as f:
        f.write("%s\nMean:%.3f" % (str(accs), np.mean(accs)))

def evaluate_ucf50_fusion():
    """ Run 5-fold cross-validation on the UCF50 dataset """
    accs = np.zeros(3)
    ob_suffix = '-max.feat.npy.gz'
    fv_suffix = '_fv.npy.gz'
    ob_root = '/home/syq/research_final/data/features/ob_ucf50_pooled_python/'
    fv_root = '/home/syq/research_final/data/features/fv_ucf50_python/'
    fv_groups, full, sets = utility.split_data(fv_root,
                                                suffix=fv_suffix,
                                                useLooCV=False)

    ob_groups, _, _ = utility.split_data(ob_root,
                                          suffix=ob_suffix,
                                          useLooCV=False)
    weights = [i / 20.0 for i in range(8, 13)]
    acc_to_weights = {}

    for weight in weights:
        print "Weight: %.2f" % weight
        accs = np.zeros(2)
        for i in xrange(2):
            ts = time.time()
            Dtrain_fv, Dtest_fv, Ytrain, Ytest = utility.load_groups(
                                                    fv_groups, np.setdiff1d(full, sets[i]),
                                                    sets[i], scale=False, verbose=False)

            Dtrain_ob, Dtest_ob, Ytrain, Ytest = utility.load_groups(
                                                    ob_groups, np.setdiff1d(full, sets[i]),
                                                    sets[i], scale=False, verbose=False)

            fv_clf = OneVsRestClassifier(estimator=LinearSVC(C=100), n_jobs=8)

            ob_clf = OneVsRestClassifier(estimator=SVC(C=10,
                                                    cache_size=1000,
                                                    kernel='linear',
                                                    probability=True),
                                        n_jobs=-1)

            # Get probabilities for late fusion
            Dtrain_fv = fv_clf.fit(Dtrain_fv, Ytrain).decision_function(Dtrain_fv)
            Dtrain_ob = ob_clf.fit(Dtrain_ob, Ytrain).decision_function(Dtrain_ob)
            Dtest_fv = fv_clf.decision_function(Dtest_fv)
            Dtest_ob = ob_clf.decision_function(Dtest_ob)

            # Scale decision values b/w 0 and 1
            Dtrain_fv = preprocessing.normalize(Dtrain_fv)
            Dtrain_ob = preprocessing.normalize(Dtrain_ob)
            Dtest_fv = preprocessing.normalize(Dtest_fv)
            Dtest_ob = preprocessing.normalize(Dtest_ob)

            # weighted averaging
            scores_train = (Dtrain_fv * weight) + (Dtrain_ob * (1 - weight))
            latefusion_clf = OneVsRestClassifier(estimator=LinearSVC(C=100), n_jobs=-1)
            latefusion_clf.fit(scores_train, Ytrain)

            scores_test = (Dtest_fv * weight) + (Dtest_ob * (1 - weight))
            latefusion_acc = latefusion_clf.score(scores_test, Ytest)
            print 'Fold', i, 'late fusion acc', latefusion_acc
            print "Train & testing time %.3f" % (time.time() - ts)
            accs[i] = latefusion_acc

    print "Mean accuracy: %.3f" % accs.mean()
    with open("fv_ucf50_accs_5fold_gridsearch.txt", "w") as f:
        for weight, accs in acc_to_weights.items():
            f.write(str(weight) + str(accs) + '\n')
    return acc_to_weights

def evaluate_hmdb51():
    """ Evaluate hmdb51 using the 3 train-test splits """
    accs = np.zeros(3)
    fv_root = '/home/syq/research_final/data/dense-traj/fv_hmdb51_python/'
    fv_suffix = '_fv.npy.gz'
    hmdb_splits = 'testTrainMulti_7030_splits/'
    categories = os.listdir(fv_root)

    vlen = 0
    for splitnum in range(1, 4):
        ts = time.time()
        print 'Split', splitnum
        trainfiles, testfiles = hmdb51_splits.loadsplit(categories,
                                                        hmdb_splits,
                                                        splitnum)
        print 'Have %d train files' % len(trainfiles)
        print 'Have %d test files' % len(testfiles)

        if not vlen:
            fp = gzip.open(os.path.join(fv_root,'%s%s'%(trainfiles[0][0][:-4],fv_suffix)),"rb")
            vlen= len(np.load(fp))
            fp.close()
            print "Feature vector length is %d" % vlen

        Dtrain = np.zeros( (len(trainfiles),vlen), np.float32 )
        Ytrain = np.ones ( (len(trainfiles)   )) * -1000

        for fi,f in enumerate(trainfiles):
            fp = gzip.open(os.path.join(fv_root,'%s%s'%(f[0][:-4],fv_suffix)),"rb")
            Dtrain[fi][:] = np.load(fp)
            fp.close()
            Ytrain[fi] = f[1]

        Dtest = np.zeros( (len(testfiles),vlen), np.float32 )
        Ytest = np.ones ( (len(testfiles)   )) * -1000

        for fi,f in enumerate(testfiles):
            fp = gzip.open(os.path.join(fv_root,'%s%s'%(f[0][:-4],fv_suffix)),"rb")
            Dtest[fi][:] = np.load(fp)
            fp.close()
            Ytest[fi] = f[1]

        print Dtrain.shape, Ytrain.shape
        print Dtest.shape, Ytest.shape

        clf = OneVsRestClassifier(estimator=LinearSVC(C=100), n_jobs=8)
        acc = clf.fit(Dtrain, Ytrain).score(Dtest, Ytest)
        print 'Split %d accuracy: %.3f' % (splitnum, acc)
        print "Train & testing time %.3f" % (time.time() - ts)
        accs[splitnum -1] = acc

    print 'Mean accuracy is %f'%(accs.mean())
    with open('fv_hmdb51_accs.txt', 'w') as f:
        f.write("%s\nMean:%.3f" % (str(accs), np.mean(accs)))

def evaluate_hmdb51_fusion():
    """ Evaluate HMDB51 with fusion of IDT & Object bank """
    vlen = 0
    ob_suffix = '-max.feat.npy.gz'
    fv_suffix = '_fv.npy.gz'
    ob_root = '/home/syq/research_final/data/features/ob_hmdb51_pooled_python/'
    fv_root = '/home/syq/research_final/data/dense-traj/fv_hmdb51_python/'
    hmdb_splits = 'testTrainMulti_7030_splits/'
    categories = os.listdir(fv_root)
    weight = 1.0
    weights = [i / 20.0 for i in range(21)]
    acc_to_weights = {}

    for weight in weights:
        print "Weight: %.2f" % weight
        accs = np.zeros(3)
        for splitnum in range(1,4):
            ts = time.time()
            trainfiles, testfiles = hmdb51_splits.loadsplit(categories,
                                                            hmdb_splits,
                                                            splitnum)
            print 'Have %d train files' % len(trainfiles)
            print 'Have %d test files' % len(testfiles)

            if not vlen:
                fp = gzip.open(os.path.join(ob_root,'%s%s'%(trainfiles[0][0][:-4],
                                                            ob_suffix)),"rb")
                vlen_ob = len(np.load(fp))
                fp.close()
                print "OB vector length is %d" % vlen_ob
                fp = gzip.open(os.path.join(fv_root,'%s%s'%(trainfiles[0][0][:-4],
                                                            fv_suffix)),"rb")
                vlen_fv = len(np.load(fp))
                fp.close()
                print "IDTFV vector length is %d" % vlen_fv

            Dtrain_ob = np.zeros( (len(trainfiles),vlen_ob), np.float32 )
            Dtrain_fv = np.zeros( (len(trainfiles),vlen_fv), np.float32 )

            Ytrain = np.ones ( (len(trainfiles)   )) * -1000

            for fi,f in enumerate(trainfiles):
                fp = gzip.open(os.path.join(ob_root,'%s%s'%(f[0][:-4],
                                                            ob_suffix)),"rb")
                Dtrain_ob[fi][:] = np.load(fp)
                fp.close()
                Ytrain[fi] = f[1]

                fp = gzip.open(os.path.join(fv_root,'%s%s'%(f[0][:-4],
                                                            fv_suffix)),"rb")
                Dtrain_fv[fi][:] = np.load(fp)
                fp.close()

            Dtest_ob = np.zeros( (len(testfiles),vlen_ob), np.float32 )
            Dtest_fv = np.zeros( (len(testfiles),vlen_fv), np.float32 )

            Ytest = np.ones ( (len(testfiles)   )) * -1000

            for fi,f in enumerate(testfiles):
                fp = gzip.open(os.path.join(ob_root,'%s%s'%(f[0][:-4],
                                                            ob_suffix)),"rb")
                Dtest_ob[fi][:] = np.load(fp)
                fp.close()
                Ytest[fi] = f[1]

                fp = gzip.open(os.path.join(fv_root,'%s%s'%(f[0][:-4],
                                                            fv_suffix)),"rb")
                Dtest_fv[fi][:] = np.load(fp)
                fp.close()

            """
            Early fusion
            Dtrain = np.hstack((Dtrain_ob, Dtrain_fv))
            Dtest = np.hstack((Dtest_ob, Dtest_fv))

            clf = OneVsRestClassifier(estimator=LinearSVC(C=100), n_jobs=8)
            acc = clf.fit(Dtrain, Ytrain).score(Dtest, Ytest)
            """
            fv_clf = OneVsRestClassifier(estimator=LinearSVC(C=100), n_jobs=8)

            ob_clf = OneVsRestClassifier(estimator=SVC(C=10,
                                                    cache_size=1000,
                                                    kernel='linear',
                                                    probability=True),
                                        n_jobs=-1)

            # Get probabilities for late fusion
            Dtrain_fv = fv_clf.fit(Dtrain_fv, Ytrain).decision_function(Dtrain_fv)
            Dtrain_ob = ob_clf.fit(Dtrain_ob, Ytrain).decision_function(Dtrain_ob)
            Dtest_fv = fv_clf.decision_function(Dtest_fv)
            Dtest_ob = ob_clf.decision_function(Dtest_ob)

            # Scale decision values b/w 0 and 1
            Dtrain_fv = preprocessing.normalize(Dtrain_fv)
            Dtrain_ob = preprocessing.normalize(Dtrain_ob)
            Dtest_fv = preprocessing.normalize(Dtest_fv)
            Dtest_ob = preprocessing.normalize(Dtest_ob)

            # Late fusion
            scores_train = (Dtrain_fv * weight) + (Dtrain_ob * (1 - weight))
            latefusion_clf = OneVsRestClassifier(estimator=LinearSVC(C=100), n_jobs=-1)
            latefusion_clf.fit(scores_train, Ytrain)

            scores_test = (Dtest_fv * weight) + (Dtest_ob * (1 - weight))
            acc = latefusion_clf.score(scores_test, Ytest)
            print 'Fold', splitnum, 'late fusion acc', acc
            print "Train & testing time %.3f" % (time.time() - ts)
            accs[splitnum-1] = acc
        acc_to_weights[weight] = accs

    print "Mean accuracy: %.3f" % accs.mean()
    with open("hmdb51_weight_gridsearch.txt", "w") as f:
        for weight, accs in acc_to_weights.items():
            f.write(str(weight) + str(accs) + '\n')
    return acc_to_weights

    """
    with open('fv_hmdb51_accs.txt', 'w') as f:
        f.write("%s\nMean:%.3f" % (str(accs), np.mean(accs)))
    """

if __name__ == '__main__':
    #evaluate_hmdb51()
    fusion_dict = evaluate_ucf50_fusion()
    #fusion_dict = evaluate_hmdb51_fusion()
