from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
from sklearn.utils.multiclass import unique_labels
from classifiers import *


class train_testPrecision_Predict(Algorithm):
    def __init__(self):
        super().__init__()

    def train(self, model=None, transform=False):
        if transform:
            self.X_train = self.scale(self.X_train)

        # train the Logistic Regression
        if model == 'LR':
            print('Training the Logistic Regression ... \n')
            self.lrcv.fit(self.X_train, self.y_train)

        # train the neural network
        elif model == 'NN':
            print('Training the Neural Network ... \n')
            self.nn.fit(self.X_train, self.y_train)

        # train the Linear Discriminant Analysis
        elif model == 'LDA':
            print('Training the Linear Discriminant Analysis ... \n')
            self.lda.fit(self.X_train, self.y_train)

        # train the  K Neighbors Classifier
        elif model == 'KNN':
            print(' Training the K Neighbors Classifier ... \n')
            self.knn.fit(self.X_train, self.y_train)

        # train the Decision Tree Classifier
        elif model == 'DTC':
            print(' Training the Decision Tree Classifier ... \n')
            self.dtc.fit(self.X_train, self.y_train)

        # train the Support Vector Machine
        elif model == 'SVM':
            print(' Training the Support Vector Machine ... \n')
            self.svm.fit(self.X_train, self.y_train)

        else:
            print('Training all the classifiers ... \n')
            print('Training the Logistic Regression ... \n')
            self.lrcv.fit(self.X_train, self.y_train)
            print('Training the Neural Network ... \n')
            self.nn.fit(self.X_train, self.y_train)
            print('Training the Linear Discriminant Analysis ... \n')
            self.lda.fit(self.X_train, self.y_train)
            print(' Training the K Neighbors Classifier ... \n')
            self.knn.fit(self.X_train, self.y_train)
            print(' Training the Decision Tree Classifier ... \n')
            self.dtc.fit(self.X_train, self.y_train)
            print(' Training the Support Vector Machine ... \n')
            self.svm.fit(self.X_train, self.y_train)



    def testPrecision(self, model=None, transform=False):
        if transform:
            self.X_test = self.scale(self.X_test)

        precision = {}
        if model == 'LR':
            precision['PRECISION LR: '] = self.lrcv.score(self.X_test, self.y_test)
        elif model == 'NN':
            precision['PRECISION NN: '] = self.nn.score(self.X_test, self.y_test)
        elif model == 'LDA':
            precision['PRECISION LDA: '] = self.lda.score(self.X_test, self.y_test)
        elif model == 'KNN':
            precision['PRECISION KNN: '] = self.knn.score(self.X_test, self.y_test)
        elif model == 'DTC':
            precision['PRECISION DTC: '] = self.dtc.score(self.X_test, self.y_test)
        elif model == 'SVM':
            precision['PRECISION SVM: '] = self.svm.score(self.X_test, self.y_test)
        else:
            print('Precision of all the classifiers ... \n')
            precision['PRECISION LR: '] = self.lrcv.score(self.X_test, self.y_test)
            precision['PRECISION NN: '] = self.nn.score(self.X_test, self.y_test)
            precision['PRECISION LDA: '] = self.lda.score(self.X_test, self.y_test)
            precision['PRECISION KNN: '] = self.knn.score(self.X_test, self.y_test)
            precision['PRECISION DTC: '] = self.dtc.score(self.X_test, self.y_test)
            precision['PRECISION SVM: '] = self.svm.score(self.X_test, self.y_test)

        return precision


    def predict(self, model=None, transform=False):
        '''

        :param model:
        :param transform:
        :return:
        '''
        if transform:
            self.X_unknown = self.scale(self.X_unknown)

        prediction = {}

        # prediction of  the Logistic Regression
        if model == 'LR':
            prediction[str('LR PREDICTION')] = ''
            for n in range(len(self.X_unknown)):
                class_index = self.lrcv.predict([self.X_unknown.iloc[n]])
                prediction[str('x[')+str(n)+str(']')+str(self.id[n])] = self.className[class_index[0]]

        # prediction of the neural network
        elif model == 'NN':
            prediction[str('NN PREDICTION')] = ''
            for n in range(len(self.X_unknown)):
                class_index = self.nn.predict([self.X_unknown.iloc[n]])
                prediction[str('x[')+str(n)+str(']')+str(self.id[n])] = self.className[class_index[0]]

        # prediction of the Linear Discriminant Analysis
        elif model == 'LDA':
            prediction[str('LDA PREDICTION')] = ''
            for n in range(len(self.X_unknown)):
                class_index = self.lda.predict([self.X_unknown.iloc[n]])
                prediction[str('x[')+str(n)+str(']')+str(self.id[n])] = self.className[class_index[0]]

        # prediction of the  K Neighbors Classifier
        elif model == 'KNN':
            prediction[str('KNN PREDICTION')] = ''
            for n in range(len(self.X_unknown)):
                class_index = self.knn.predict([self.X_unknown.iloc[n]])
                prediction[str('x[')+str(n)+str(']')+str(self.id[n])] = self.className[class_index[0]]

        # prediction of the Decision Tree Classifier
        elif model == 'DTC':
            prediction[str('DTC PREDICTION')] = ''
            for n in range(len(self.X_unknown)):
                class_index = self.dtc.predict([self.X_unknown.iloc[n]])
                prediction[str('x[')+str(n)+str(']')+str(self.id[n])] = self.className[class_index[0]]

        # prediction of the Support Vector Machine
        elif model == 'SVM':
            prediction[str('SVM PREDICTION')] = ''
            for n in range(len(self.X_unknown)):
                class_index = self.svm.predict([self.X_unknown.iloc[n]])
                prediction[str('x[')+str(n)+str(']')+str(self.id[n])] = self.className[class_index[0]]
        return prediction

    def information(self):
        """
        calcul the confusion matrix and the classification report
        :param model: model to use
        :return: confusion matrice and classification report
        """

        #cm = np.zeros((99, 99))
        #cr = 0.0
        #if model == 'LR':
        self.lda.fit(self.X_train.iloc[0:5, :], self.y_train[0:5])
        predictions = self.lda.predict(self.X_test.iloc[0:5, :])
        cm = confusion_matrix(self.y_test[0:5], predictions)
        cr = classification_report(self.y_test[0:5], predictions)
        return cm, cr, predictions

    def plot_confusion_matrix(self, normalize=False, title=None):
        """
        This function prints and plots the confusion matrix.
         Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'
        classes = self.y_test[0:5]
        # Compute confusion matrix
        cm, cr, predictions = self.information()

        # Only use the labels that appear in the data
        classes = classes[unique_labels(self.y_test[0:5], predictions)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
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
        return ax


    def submission(self):
        """
        compute the probability for a sample to be in a class
        :return:
        """
        self.lrcv.fit(self.X_train, self.y_train)
        prediction = self.lrcv.predict_proba(self.X_unknown)
        submission = pd.DataFrame(prediction, columns=self.className)
        submission.insert(0, 'id', self.id)
        submission.reset_index()
        return submission
