from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.utils import class_weight

from fake_news_classifier.util import log


def plot_keras_history(training_history, with_validation):
    # Accuracy Plot
    plt.plot(training_history.history['acc'])
    if with_validation:
        plt.plot(training_history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # Loss Plot
    plt.plot(training_history.history['loss'])
    if with_validation:
        plt.plot(training_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


# The below is from SKLearn Docs
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        log("Normalized confusion matrix")
    else:
        log('Confusion matrix, without normalization')

    log(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()


# Returns useful metrics in a dict, logs if asked
# Accuracy, F1 Micro, F1 Macro, F1 Weighted, Confusion matrix
def eval_predictions(y_true, y_pred, classes, print_results=False):
    # Plot confusion matrix
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        normalize=True,
        classes=classes
    )
    # TODO: Precision, recall, return results in a dict
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    f1_score_micro = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_score_macro = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    f1_score_weighted = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    if print_results:
        log("Prediction Evaluation", header=True)
        log(f"Accuracy: {accuracy}")
        log(f"F1 Score (Macro): {f1_score_macro}")
        log(f"F1 Score (Micro): {f1_score_micro}")
        log(f"F1 Score (Weighted): {f1_score_weighted}")


# K-Fold Validation - returns a list of fold objects of length k: (train_idx, test_idx)
def k_fold_indicies(x, y, k=10):
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    return skf.split(x, y)


# Computes class weights for keras with to_categorical applied to y-data
def get_class_weights(y_train):
    y_ints = categorical_to_idx(y_train)
    return class_weight.compute_class_weight(
        'balanced',
        np.unique(y_ints),
        y_ints
    )


# Returns Keras 2D categorical arrays (ex. [[0 0 1 0]] to an integer array: [3])
def categorical_to_idx(cat_labels):
    return [np.argmax(y) for y in cat_labels]
