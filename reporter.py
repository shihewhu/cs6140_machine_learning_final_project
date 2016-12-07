#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from activity_recongnizer import ActivityRecognizer


class ExperimentBase(object):
    def set_up(self, x_axis, y_axis):
        self.x_axis = x_axis
        self.y_axis = y_axis

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ind = np.arange(len(self.x_axis))
        width = 0.15
        y1, y2, y3, y4 = self.y_axis
        rects1 = ax.bar(ind, y1, width, color='red')
        rects2 = ax.bar(ind + width, y2, width, color='green')
        rects3 = ax.bar(ind + 2 * width, y3, width, color='black')
        rects4 = ax.bar(ind + 3 * width, y4, width, color='yellow')
        ax.set_xlim(-width, len(ind) + width)
        ax.set_ylim((0, 1.5))
        ax.set_ylabel('error')
        ax.set_xticks(ind + width)
        xtickNames = ax.set_xticklabels(self.x_axis)
        plt.setp(xtickNames, rotation=45, fontsize=10)
        ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]),
                  ('Activity Model Training Error', 'Activity Model Test Error', 'Location Model Training Error', 'Location Model Test Error'))
        plt.show()


class CompareModelExperiment(ExperimentBase):
    def do_experiment(self):
        """This experiment will take very long time. You can also train 3 models seperately"""
        print "Train NN"
        ar = ActivityRecognizer(25, 14, 'training_data')
        ar.train("NN", GPS_only=False, solver='lbfgs', hidden_layer_sizes=(5, 2, 2, 2))
        print "Test NN"
        nn_training_error1, nn_training_error2 = ar.test('training_data')
        nn_test_error1, nn_test_error2 = ar.test('test_data')
        print "Train SVM"
        ar = ActivityRecognizer(25, 14, 'training_data')
        ar.train("SVM", GPS_only=False, C=10, kernel='linear')
        print "Test SVM"
        svm_training_error1, svm_training_error2 = ar.test('training_data')
        svm_test_error1, svm_test_error2 = ar.test('test_data')
        print "Train RF"
        print nn_training_error1, svm_training_error1
        print nn_training_error2, svm_training_error2
        print nn_test_error1, svm_test_error1
        print nn_test_error2, svm_test_error2
        exit(0)
        ar = ActivityRecognizer(25, 14, 'training_data')
        ar.train("RF", GPS_only=False, n_estimators=2000, max_depth=1)
        print "Test RF"
        rf_training_error1, rf_training_error2 = ar.test('training_data')
        rf_test_error1, rf_test_error2 = ar.test('test_data')
        self.set_up(
            ["NN", "SVM", "RF"],
            (
                [nn_training_error1, svm_training_error1, rf_training_error1],
                [nn_test_error1, svm_test_error1, rf_test_error1],
                [nn_training_error2, svm_training_error2, rf_training_error2],
                [nn_test_error2, svm_test_error2, rf_training_error2],
            ),
        )
        self.plot()


class NNExperiment(ExperimentBase):
    def do_experiment(self):
        training_error1 = []
        test_error1 = []
        training_error2 = []
        test_error2 = []
        for h in [(5), (5, 2), (5, 2, 2), (5, 3, 2), (5, 2, 2, 2)]:
            ar = ActivityRecognizer(25, 14, 'training_data')
            ar.train("NN", GPS_only=False, solver='lbfgs', hidden_layer_sizes=h)
            print "Test NN"
            nn_training_error1, nn_training_error2 = ar.test('training_data')
            nn_test_error1, nn_test_error2 = ar.test('test_data')
            training_error1.append(nn_training_error1)
            training_error2.append(nn_training_error2)
            test_error1.append(nn_test_error1)
            test_error2.append(nn_test_error2)
        self.set_up(["(5)", "(5, 2)", "(5, 2, 2)", "(5, 3, 2)", "(5, 2, 2, 2)"], (training_error1, test_error1, training_error2, test_error2))
        self.plot()


if __name__ == "__main__":
    e = CompareModelExperiment()
    e.do_experiment()
    e = NNExperiment()
    e.do_experiment()
