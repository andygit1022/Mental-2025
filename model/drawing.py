import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import itertools


class DrawPlot:
    def __init__(self, fn):
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.f1 = []
        self.f1_0 = []
        self.f1_1 = []
        self.val_f1_0 = []
        self.val_f1_1 = []
        self.logs = []
        self.legend = False
        self.best_acc = 0
        self.best_f1_0 = 0
        self.best_f1_1 = 0
        self.best_pr0 = 0
        self.best_pr1 = 0
        self.best_loss = 1
        self.best_f1 = 0
        self.val_f1 = []
        self.val_pr0 = []
        self.val_pr1 = []
        self.val_rc0 = []
        self.val_rc1 = []
        self.pr0 = []
        self.pr1 = []
        self.rc0 = []
        self.rc1 = []
        self.epoch = 0
        plt.ion()
        self.fig = plt.figure()
        self.fn = fn

    def save(self, epoch, logs, model):
        self.epoch += 1
        self.x.append(self.epoch)
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))

        self.pr0.append(logs.get('precision'))
        self.rc0.append(logs.get('recall'))
        if self.pr0[-1] + self.rc0[-1] == 0:
            f1_0 = 0
        else:
            f1_0 = 2 * self.pr0[-1] * self.rc0[-1] / (self.pr0[-1] + self.rc0[-1])
        self.f1_0.append(f1_0)

        self.pr1.append(logs.get('precision_1'))
        self.rc1.append(logs.get('recall_1'))
        self.acc.append(logs.get('categorical_accuracy'))
        if self.pr1[-1] + self.rc1[-1] == 0:
            f1_1 = 0
        else:
            f1_1 = 2 * self.pr1[-1] * self.rc1[-1] / (self.pr1[-1] + self.rc1[-1])

        f1 = (f1_0 + f1_1) * 0.5

        self.f1_1.append(f1_1)
        self.f1.append(f1)

        self.val_pr0.append(logs.get('val_precision'))
        self.val_rc0.append(logs.get('val_recall'))
        self.val_losses.append(logs.get('val_loss'))

        if self.val_pr0[-1] + self.val_rc0[-1] == 0:
            val_f1_0 = 0
        else:
            val_f1_0 = 2 * self.val_pr0[-1] * self.val_rc0[-1] / (self.val_pr0[-1] + self.val_rc0[-1])
        self.val_f1_0.append(val_f1_0)

        self.val_acc.append(logs.get('val_categorical_accuracy'))
        self.val_pr1.append(logs.get('val_precision_1'))
        self.val_rc1.append(logs.get('val_recall_1'))

        if self.val_pr1[-1] + self.val_rc1[-1] == 0:
            val_f1_1 = 0
        else:
            val_f1_1 = 2 * self.val_pr1[-1] * self.val_rc1[-1] / (self.val_pr1[-1] + self.val_rc1[-1])
        self.val_f1_1.append(val_f1_1)
        val_f1 = (val_f1_0 + val_f1_1) * 0.5
        self.val_f1.append(val_f1)
        test1 = (
            (self.val_losses[-1] < self.best_loss)
        )
        test2 = (
                (self.val_f1[-1] > self.best_f1) and
                (self.val_f1_0[-1] > self.best_f1_0) and
                (self.val_f1_1[-1] > self.best_f1_1) and
                (self.val_pr0[-1] > 0.5) and
                (self.val_pr1[-1] > 0.5)
        )
        if ((test1 or test2)):
            print(
                "\nmodel save [%s, %s]: [loss: %f]\t[f1: %f --> %f]\t[f1_0: %f --> %f]\t[f1_1: %f --> %f]\t[pr0: %f --> %f]\t[pr1: %f --> %f]" % (
                    test1, test2,
                    self.val_losses[-1],
                    self.best_f1, self.val_f1[-1],
                    self.best_f1_0, self.val_f1_0[-1], self.best_f1_1, self.val_f1_1[-1], self.best_pr0,
                    self.val_pr0[-1],
                    self.best_pr1, self.val_pr1[-1]))
            if test2:
                self.best_f1 = self.val_f1[-1]
                self.best_f1_0 = self.val_f1_0[-1]
                self.best_f1_1 = self.val_f1_1[-1]
            self.best_pr0 = self.val_pr0[-1]
            self.best_pr1 = self.val_pr1[-1]
            if test1:
                self.best_loss = self.val_losses[-1]

            model.save(self.fn)

        if self.epoch % 10 == 0:
            plt.close()
            self.legend = False
            return

        plt.subplot(1, 7, 1)  # nrows=2, ncols=1, index=1
        plt.plot(self.x, self.losses, 'b', label="loss")
        plt.plot(self.x, self.val_losses, 'r', label="val_loss")
        if not self.legend:
            plt.legend(loc='upper left')
        plt.subplot(1, 7, 2)  # nrows=2, ncols=1, index=1
        plt.plot(self.x, self.f1, 'b', label="f1")

        plt.plot(self.x, self.val_f1, 'r', label="val_f1")
        plt.ylim([0, 1])
        if not self.legend:
            plt.legend(loc='upper left')
        plt.subplot(1, 7, 3)  # nrows=2, ncols=1, index=1
        plt.plot(self.x, self.f1_0, 'b', label="f1_0")

        plt.plot(self.x, self.val_f1_0, 'r', label="val_f1_0")
        plt.ylim([0, 1])
        if not self.legend:
            plt.legend(loc='upper left')
        plt.subplot(1, 7, 4)  # nrows=2, ncols=1, index=1
        plt.plot(self.x, self.f1_1, 'b', label="f1_1")

        plt.plot(self.x, self.val_f1_1, 'r', label="val_f1_1")
        plt.ylim([0, 1])
        if not self.legend:
            plt.legend(loc='upper left')
        plt.subplot(1, 7, 5)  # nrows=2, ncols=1, index=1
        plt.plot(self.x, self.pr0, 'b', label="pr0")

        plt.plot(self.x, self.val_pr0, 'r', label="val_pr0")
        plt.ylim([0, 1])
        if self.legend == False:
            plt.legend(loc='upper left')
        plt.subplot(1, 7, 6)  # nrows=2, ncols=1, index=1
        plt.plot(self.x, self.pr1, 'b', label="pr1")

        plt.plot(self.x, self.val_pr1, 'r', label="val_pr1")
        plt.ylim([0, 1])
        if self.legend == False:
            plt.legend(loc='upper left')
        plt.subplot(1, 7, 7)  # nrows=2, ncols=1, index=1
        plt.plot(self.x, self.acc, 'b', label="acc")

        plt.plot(self.x, self.val_acc, 'r', label="val_acc")
        plt.ylim([0, 1])
        if self.legend == False:
            plt.legend(loc='upper left')
            self.legend = True
        plt.pause(0.01)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.ioff()
    # plt.ion()
    plt.close()
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
