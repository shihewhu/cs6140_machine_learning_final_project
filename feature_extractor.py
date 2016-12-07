#!/usr/bin/env python
# encoding: utf-8
import csv
import os
import numpy as np
import scipy.stats as sp
from geopy.distance import great_circle
from datetime import datetime
from time import mktime
from enum import Enum


class Feature(Enum):
    TIMESTAMP = 0
    AVG_LAT = 1,
    AVG_LNG = 2,
    GPS_ACCURACY = 3,
    AVG_SPEED_BY_GPS = 4,
    ACCLERATOR_STD = 5,
    MAX_ACCLERATOR = 6,
    MIN_ACCLERATOR = 7,
    PERCENTILE_25_ACCLERATOR = 10,
    PERCENTILE_75_ACCLERATOR = 11,
    ACCLERATOR_ENTROPY = 12,
    AVG_ROLL = 13,
    AVG_YAW = 14,
    AVG_PITCH = 15,
    WEEK_DAY = 16,
    DISTANCE = 17


class FeatureExtractor(object):
    def __init__(self, raw_data_dir, window_size=60, step=30):
        self.raw_data_dir = raw_data_dir
        self.raw_data = None
        self.window_size = window_size
        self.step = step
        self.features_list = []
        self.start_time = None
        self.end_time = None

    def read_raw_data(self, filename):
        raw_data = {}
        # walk = os.walk(self.raw_data_dir)
        # root, _, filenames = next(walk)
        # for filename in filenames:
        # print filename
        path = os.path.join(self.raw_data_dir, filename)
        with open(path, 'rb') as f:
            reader = csv.reader(f)
            next(reader)
            for data_point in reader:
                (timestamp, data_source, data_type, value) = data_point
                if not self.start_time:
                    self.start_time = int(mktime(datetime.strptime(timestamp[:-4], '%Y-%m-%dT%H:%M').timetuple()))
                unix_time = mktime(datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%SZ').timetuple())
                self.end_time = int(unix_time)
                raw_data.setdefault(unix_time, {}).setdefault(data_source.strip(' '), {})[data_type.strip(' ')] = value.strip(' ')
        self.raw_data = raw_data

    def _get_geo_distance(self, start, dest):
        return great_circle(start, dest).meters

    def _get_averge_speed_by_gps(self, start, dest, duration):
        return self._get_geo_distance(start, dest) / duration

    def extract(self):
        print self.start_time, self.end_time
        if self.raw_data is None:
            raise Exception("You should read data first")
        for window_start_time in xrange(self.start_time, self.end_time, self.step):
            window_end_time = window_start_time + self.window_size
            data_in_window = [self.raw_data[time] for time in range(window_start_time, window_end_time) if self.raw_data.get(time)]
            extract_functions = [extract_function for extract_function in dir(FeatureExtractor) if "EXTRACT" in extract_function]
            self.acclerator_vector = self._build_total_acclerator_vector(data_in_window)
            feature_vector = [getattr(self, extract_function)(data_in_window, window_start_time, window_end_time) for extract_function in extract_functions]
            feature_vector = sorted(feature_vector, key=lambda x: x[0].value)
            self.features_list.append(feature_vector)
            # set start_time to None for next read
            self.start_time = None

    def write_features_to_csv(self, dir_name, filename):
        # header = ['TIMESTAMP', 'AVG_LAT', 'AVG_LNG', 'DISTANCE']
        header = [e.name for e in Feature]
        if dir_name not in os.listdir('.'):
            os.mkdir(dir_name)
        path = os.path.join(dir_name, filename)
        with open(path, 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for features in self.features_list:
                writer.writerow([feature[1] for feature in features])

    def EXTRACT_timestamp(self, data_in_window, window_start_time, window_end_time):
        return (Feature.TIMESTAMP, window_start_time)

    def EXTRACT_average_lat(self, data_in_window, window_start_time, window_end_time):
        vector = np.array([float(data_point['GPS']['Latitude']) for data_point in data_in_window if data_point.get('GPS')])
        return (Feature.AVG_LAT, np.mean(vector))

    def EXTRACT_average_lng(self, data_in_window, window_start_time, window_end_time):
        vector = np.array([float(data_point['GPS']['Longitude']) for data_point in data_in_window if data_point.get('GPS')])
        return (Feature.AVG_LNG, np.mean(vector))

    def EXTRACT_distance(self, data_in_window, window_start_time, window_end_time):
        src_point, dst_point = None, None
        for data_point in data_in_window:
            if data_point.get('GPS'):
                src_point = (float(data_point['GPS']['Longitude']), float(data_point['GPS']['Latitude']))
                break
        for data_point in reversed(data_in_window):
            if data_point.get('GPS'):
                dst_point = (float(data_point['GPS']['Longitude']), float(data_point['GPS']['Latitude']))
                break
        return (Feature.DISTANCE, self._get_geo_distance(src_point, dst_point))

    def EXTRACT_average_speed_by_gps(self, data_in_window, window_start_time, window_end_time):
        src_point, dst_point = None, None
        for data_point in data_in_window:
            if data_point.get('GPS'):
                src_point = (float(data_point['GPS']['Longitude']), float(data_point['GPS']['Latitude']))
                break
        for data_point in reversed(data_in_window):
            if data_point.get('GPS'):
                dst_point = (float(data_point['GPS']['Longitude']), float(data_point['GPS']['Latitude']))
                break
        return (Feature.AVG_SPEED_BY_GPS, self._get_averge_speed_by_gps(src_point, dst_point, self.window_size))

    def EXTRACT_average_yaw(self, data_in_window, *args, **kwargs):
        vector = np.array([float(data_point['Attitude']['Yaw']) for data_point in data_in_window if data_point.get('Attitude')])
        return (Feature.AVG_YAW, np.mean(vector))

    def EXTRACT_average_roll(self, data_in_window, *args, **kwargs):
        vector = np.array([float(data_point['Attitude']['Roll']) for data_point in data_in_window if data_point.get('Attitude')])
        return (Feature.AVG_ROLL, np.mean(vector))

    def EXTRACT_average_pitch(self, data_in_window, *args, **kwargs):
        vector = np.array([float(data_point['Attitude']['Pitch']) for data_point in data_in_window if data_point.get('Attitude')])
        return (Feature.AVG_PITCH, np.mean(vector))

    def EXTRACT_gps_accuracy(self, data_in_window, *args, **kwargs):
        vector = np.array([float(data_point['GPS']['Horizontal Accuracy']) for data_point in data_in_window if data_point.get('GPS')])
        return (Feature.GPS_ACCURACY, np.mean(vector))

    @staticmethod
    def _build_total_acclerator_vector(data_in_window):
        x_vector = np.array([float(data_point['Acceleration (via User)']['x']) for data_point in data_in_window if data_point.get('Acceleration (via User)')])
        y_vector = np.array([float(data_point['Acceleration (via User)']['y']) for data_point in data_in_window if data_point.get('Acceleration (via User)')])
        z_vector = np.array([float(data_point['Acceleration (via User)']['z']) for data_point in data_in_window if data_point.get('Acceleration (via User)')])
        return np.sum(np.square(np.vstack((x_vector, y_vector, z_vector))), axis=0)

    def EXTRACT_max_acclerator(self, data_in_window, *args, **kwargs):
        feature = (Feature.MAX_ACCLERATOR, np.amax(self.acclerator_vector)) if len(self.acclerator_vector) > 0 else (Feature.MAX_ACCLERATOR, 0.0)
        return feature

    def EXTRACT_min_acclerator(self, data_in_window, *args, **kwargs):
        feature = (Feature.MIN_ACCLERATOR, np.amin(self.acclerator_vector)) if len(self.acclerator_vector) > 0 else (Feature.MIN_ACCLERATOR, 0.0)
        return feature

    def EXTRACT_acclerator_std(self, data_in_window, *args, **kwargs):
        feature = (Feature.ACCLERATOR_STD, np.std(self.acclerator_vector)) if len(self.acclerator_vector) > 0 else (Feature.ACCLERATOR_STD, 0.0)
        return feature

    def EXTRACT_percentile_25_acclerator(self, data_in_window, *args, **kwargs):
        feature = (Feature.PERCENTILE_25_ACCLERATOR, np.percentile(self.acclerator_vector, 25)) if len(self.acclerator_vector) > 0 else (Feature.PERCENTILE_25_ACCLERATOR, 0.0)
        return feature

    def EXTRACT_percentile_75_acclerator(self, data_in_window, *args, **kwargs):
        feature = (Feature.PERCENTILE_75_ACCLERATOR, np.percentile(self.acclerator_vector, 75)) if len(self.acclerator_vector) > 0 else (Feature.PERCENTILE_75_ACCLERATOR, 0.0)
        return feature

    def EXTRACT_acclerator_entory(self, data_in_window, *args, **kwargs):
        if len(self.acclerator_vector) < 1:
            return (Feature.ACCLERATOR_ENTROPY, 0.0)
        total_acclerator_vector = np.array([format(v, '.2f') for v in self.acclerator_vector])
        unique, counts = np.unique(total_acclerator_vector, return_counts=True)
        counter = dict(zip(unique, counts / float(np.shape(total_acclerator_vector)[0])))
        entropy = sp.entropy(np.array([counter[a] for a in total_acclerator_vector]))
        return (Feature.ACCLERATOR_ENTROPY, entropy)

    def EXTRACT_week_day(self, data_in_window, window_start_time, window_end_time):
        return (Feature.WEEK_DAY, datetime.fromtimestamp(window_start_time).strftime('%w'))

if __name__ == "__main__":
    walk = os.walk('clean_data')
    root, _, filenames = next(walk)
    for name in filenames:
        print name
        feature_extractor = FeatureExtractor('clean_data')
        # feature_extractor.read_raw_data('Gathered-Recording-2016-11-10-08-01-42.csv')
        feature_extractor.read_raw_data(name)
        feature_extractor.extract()
        feature_extractor.write_features_to_csv('features', name)
