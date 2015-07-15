from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import os

# Compute Precision-Recall and plot curve
#pylab inline
def plot_precision_recall_curve(y_true, y_score, title, categories):
    precision = dict()
    recall = dict()
    average_precision = dict()
    n_classes = len(categories)

    if y_true.ndim <= 1:
        lb = LabelBinarizer()
        y_true = lb.fit_transform(y_true)

    for i in range(0, n_classes):
        precision[i],  recall[i], _ = precision_recall_curve(y_true[:, i],
                                                             y_score[:, i])
        average_precision[i] =  average_precision_score(y_true[:, i],
                                                        y_score[:, i])

    # Compute average ROC curve and ROC area
    precision["avg"], recall["avg"], _ = precision_recall_curve(y_true.ravel(),
                                                                y_score.ravel())
    average_precision["avg"] = average_precision_score(y_true, y_score,
                                                       average="macro")

    # Plot Precision-Recall curve for each class
    plt.clf()
    ax = plt.subplot(111)
    ax.plot(recall["avg"], precision["avg"],
            label='average Precision-recall curve (area = {0:0.2f})'
            ''.format(average_precision["avg"]))
    for i in range(n_classes):
        ax.plot(recall[i], precision[i],
                label='{0} (area = {1:0.2f})'
                ''.format(categories[i], average_precision[i]))

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.show()
    return average_precision

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, categories, title="Confusion Matrix"):
    if y_true.ndim > 1:
        nclasses = y_pred.shape[1]
        cm = np.zeros((nclasses, nclasses), np.int8)
        for i in range(y_true.shape[0]):
            pred_indices = np.where(y_pred[i, :] == 1)[0]
            true_indices = np.where(y_true[i, :] == 1)[0]
            for j in pred_indices:
                if j == 9:
                    print true_indices, pred_indices
                for k in true_indices:
                    cm[k, j] += 1

    else:
        cm = confusion_matrix(y_true, y_pred)
    norm_conf = []
    for i in cm:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure(figsize=(14,14))
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.Blues,
                    interpolation='nearest')

    width = len(cm)
    height = len(cm[0])

    for x in xrange(width):
        for y in xrange(height):
            if(cm[x][y] != 0):
                ax.annotate(str(cm[x][y]), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center')

    cb = fig.colorbar(res)

    plt.xticks(range(width), categories[:width], rotation=90)
    plt.yticks(range(height), categories[:height])
#    plt.title('Confusion matrix for Object Bank on CCV')
    plt.show()
    return cm
