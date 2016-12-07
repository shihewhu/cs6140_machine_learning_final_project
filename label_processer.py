#!/usr/bin/env python
# encoding: utf-8
import csv
import os
from datetime import datetime
from time import mktime


class LabelWorker(object):
    def __init__(self, label_filename, step=30):
        self.label_filename = label_filename
        self.labels = self._read_labels()
        self.step = step

    def _read_labels(self):
        activity_label_name_set = set()
        loaction_label_name_set = set()
        labels = []
        with open(self.label_filename, 'rb') as f:
            reader = csv.reader(f)
            next(reader)
            for d in reader:
                labels.append(d)
                activity_label_name_set.add(d[2])
                loaction_label_name_set.add(d[3])
            self.activity_label_name_with_idx = {label_name: idx for idx, label_name in enumerate(list(activity_label_name_set))}
            self.loaction_label_name_with_idx = {label_name: idx for idx, label_name in enumerate(list(loaction_label_name_set))}
        return labels

    def write_label_name_map(self, path1, path2):
        header = ['label_name', 'index']
        activity_label_name_with_idx_tuple = sorted(self.activity_label_name_with_idx.items(), key=lambda x: x[1])
        loaction_label_name_with_idx_tuple = sorted(self.loaction_label_name_with_idx.items(), key=lambda x: x[1])
        with open(path1, 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for tuple in activity_label_name_with_idx_tuple:
                label, idx = tuple
                writer.writerow([label, idx])
        with open(path2, 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for tuple in loaction_label_name_with_idx_tuple:
                label, idx = tuple
                writer.writerow([label, idx])

    @staticmethod
    def _read_features(feature_filename):
        with open(feature_filename, 'rb') as f:
            reader = csv.reader(f)
            header = next(reader)
            features = {int(d[0]): d for d in reader}
        return header, features

    def add_label_to_features(self, path):
        self.features_with_label = []
        self.feature_header, self.features = self._read_features(path)
        self.feature_header.extend(['LABEL1', 'LABEL2'])
        for label in self.labels:
            start_time = int(mktime(datetime.strptime(label[0], '%m/%d/%y %H:%M').timetuple()))
            end_time = int(mktime(datetime.strptime(label[1], '%m/%d/%y %H:%M').timetuple()))
            for time in range(start_time, end_time, self.step):
                if not self.features.get(time):
                    continue
                feature = self.features[time]
                feature.append(self.activity_label_name_with_idx[label[2]])
                feature.append(self.loaction_label_name_with_idx[label[3]])
                self.features_with_label.append(feature)

    def write_back(self, dir_name, filename):
        if dir_name not in os.listdir('.'):
            os.mkdir(dir_name)
        path = os.path.join(dir_name, filename)
        with open(path, 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(self.feature_header)
            for feature_with_label in self.features_with_label:
                writer.writerow(feature_with_label)


if __name__ == "__main__":
    walk = os.walk('features')
    root, _, filenames = next(walk)
    label_worker = LabelWorker('labels.csv')
    label_worker.write_label_name_map('activity_labels.csv', 'loaction_labels.csv')
    for name in filenames:
        path = os.path.join(root, name)
        label_worker = LabelWorker('labels.csv')
        label_worker.add_label_to_features(path)
        label_worker.write_back('training_data', name)
