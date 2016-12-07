#!/usr/bin/env python
# encoding: utf-8:w

import csv
import os
import sys
import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans


class ActivityRecognizationModel(object):
    def __init__(self, location_label_num, activity_label_num, training_data_dir):
        self.training_data_X, self.training_data_Y1, self.training_data_Y2, self.data_GPS = self.load_data(training_data_dir)
        self.location_label_num = location_label_num
        self.activity_label_num = activity_label_num
        self.activity_model = None
        self.location_model = None

    @staticmethod
    def load_data(data_dir, limit=sys.maxint):
        data_X = []
        data_GPS = []
        data_Y1 = []
        data_Y2 = []
        walk = os.walk(data_dir)
        root, _, filenames = next(walk)
        count = 0
        for name in filenames:
            if count > limit:
                break
            else:
                count += 1
            path = os.path.join(root, name)
            with open(path, 'rb') as f:
                reader = csv.reader(f)
                next(reader)
                for data in reader:
                    if 'nan' in data:
                        continue
                    data = map(float, data)
                    data_GPS.append(data[2:4])
                    data_X.append(data[:2] + data[4:-2])
                    data_Y1.append(data[-2])
                    data_Y2.append(data[-1])
        return np.array(data_X), np.array(data_Y1), np.array(data_Y2), np.array(data_GPS)

    def write_data_GPS_and_kmeans(self, output_path):
        min_max_scaler = preprocessing.MinMaxScaler()
        normalized_data = min_max_scaler.fit_transform(self.data_GPS)
        kmeans = KMeans(n_clusters=self.location_label_num, random_state=0).fit(normalized_data)
        with open(output_path, 'wb') as f:
            writer = csv.writer(f)
            header = ["lat", "lng", "Cluster Result"]
            writer.writerow(header)
            for gps, cluster in zip(self.data_GPS, kmeans.labels_):
                data = [gps[0], gps[1], cluster]
                writer.writerow(data)

    def train(self, method, GPS_only, **kwargs):
        min_max_scaler = preprocessing.MinMaxScaler()
        normalized_data = min_max_scaler.fit_transform(self.training_data_X) if not GPS_only else min_max_scaler.fit_transform(self.data_GPS)
        self.training_data_X = self._preprocess_by_KMeans(normalized_data, self.data_GPS) if not GPS_only else normalized_data
        getattr(self, "_train_by_{}".format(method))(**kwargs)

    def _preprocess_by_KMeans(self, data_X, data_GPS):
        min_max_scaler = preprocessing.MinMaxScaler()
        normalized_data = min_max_scaler.fit_transform(data_GPS)
        kmeans = KMeans(n_clusters=self.location_label_num, random_state=0).fit(normalized_data)
        category = []
        for label in kmeans.labels_:
            vector = np.array([0] * self.location_label_num)
            vector[int(label)] = 1.0
            category.append(vector)
        return np.hstack((data_X, np.array(category)))

    def _train_by_SVM(self, C, kernel):
        self.activity_model = svm.SVC(C=C, kernel=kernel, decision_function_shape='ovr')
        self.location_model = svm.SVC(C=C, kernel=kernel, decision_function_shape='ovr')
        print "training activity model"
        self.activity_model.fit(self.training_data_X, self.training_data_Y1)
        print "training location model"
        self.location_model.fit(self.training_data_X, self.training_data_Y2)

    def _train_by_NN(self, solver, hidden_layer_sizes):
        self.activity_model = MLPClassifier(solver=solver, alpha=1e-5, hidden_layer_sizes=hidden_layer_sizes, random_state=1)
        self.activity_model.fit(self.training_data_X, self.training_data_Y1)
        self.location_model= MLPClassifier(solver=solver, alpha=1e-5, hidden_layer_sizes=hidden_layer_sizes, random_state=1)
        self.location_model.fit(self.training_data_X, self.training_data_Y2)

    def _train_by_RF(self, n_estimators, max_depth):
        self.activity_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        self.activity_model.fit(self.training_data_X, self.training_data_Y1)
        self.location_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        self.location_model.fit(self.training_data_X, self.training_data_Y2)

    def predict(self, X):
        Y1 = self.activity_model.predict(X)
        Y2 = self.location_model.predict(X)
        return Y1, Y2

    def test(self, test_dir, limit=sys.maxint, use_RNN=False, GPS_only=False):
        test_X, test_Y1, test_Y2, test_data_GPS = self.load_data(test_dir, limit)
        min_max_scaler = preprocessing.MinMaxScaler()
        normalized_data = min_max_scaler.fit_transform(test_X)
        test_X = self._preprocess_by_KMeans(normalized_data, test_data_GPS) if not GPS_only else min_max_scaler.fit_transform(test_data_GPS)
        predict_Y1, predict_Y2 = self.predict(test_X) if not use_RNN else self.predict_by_RNN(test_X)
        activity_prediction_error = np.count_nonzero(predict_Y1 - test_Y1) / float(len(test_Y1))
        location_prediction_error = np.count_nonzero(predict_Y2 - test_Y2) / float(len(test_Y2))
        return activity_prediction_error, location_prediction_error


if __name__ == "__main__":
    ar = ActivityRecognizationModel(25, 14, 'training_data')
    ar.train("NN", GPS_only=False, solver='lbfgs', hidden_layer_sizes=(5, 2, 2, 2))
    print ar.test('training_data', GPS_only=False)
