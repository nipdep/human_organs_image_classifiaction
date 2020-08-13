import matplotlib.pyplot as plt
from numpy import np
from sklearn.metrics import roc_curve, auc

plt.style.use('ggplot')

from sklearn.metrics import confusion_matrix
from .cf_metrix import make_confusion_matrix


# %matplotlib inline


def acc_n_loss(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


def ROC_classes(n_classes, y_test, y_predict_proba, labels=[]):
    # Compute ROC curve and ROC AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    all_y_test_i = np.array([])
    all_y_predict_proba = np.array([])
    for i in range(n_classes):
        y_test_i = map(lambda x: 1 if x == i else 0, y_test)
        all_y_test_i = np.concatenate([all_y_test_i, y_test_i])
        all_y_predict_proba = np.concatenate([all_y_predict_proba, y_predict_proba[:, i]])
        fpr[i], tpr[i], _ = roc_curve(y_test_i, y_predict_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["average"], tpr["average"], _ = roc_curve(all_y_test_i, all_y_predict_proba)
    roc_auc["average"] = auc(fpr["average"], tpr["average"])

    # Plot average ROC Curve
    plt.figure()
    plt.plot(fpr["average"], tpr["average"],
             label='Average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["average"]),
             color='deeppink', linestyle=':', linewidth=4)

    # Plot each individual ROC curve
    if len(labels) != 0:
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], lw=2,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(labels[i], roc_auc[i]))
    else:
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], lw=2,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()


def ROC_classifies():
    pass


def plot_confusion_metrix(y, y_pred, labels):
    # Get the confusion matrix
    cf_matrix = confusion_matrix(y, y_pred)

    make_confusion_matrix(cf_matrix,
                          group_names=labels,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None)
