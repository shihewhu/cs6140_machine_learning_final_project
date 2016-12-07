#!/usr/bin/env python
# encoding: utf-8

import csv
import os

EXCLUDE_DATA_SOURCE = [
    ' Screen',
    ' Proximity',
    ' Battery',
    ' Memory',
    ' Device Orientation',
    ' Storge',
    ' Cell Radio',
    ' Altimeter (Barometer)',
    ' Magnetometer (raw)',
    ' Accelerometer (raw)',
    ' Acceleration (via Gravity)',
    ' Acceleration (total)',
    ' Compass',
    ' Storage',
    ' Bluetooth',
    ' Microphone',
    ' Gyrometer (raw)',
    ' Gyrometer (smooth)',
    ' WiFi',
]
EXCLUDE_DATA = [
    ' Enabled',
    ' Authorisation Status',
    ' Floor',
    ' Quarternion w',
    ' Quarternion x',
    ' Quarternion y',
    ' Quarternion z',
    ' Vertical Accuracy',
]


class DataCleaner(object):
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir

    def format_raw_data(self):
        walk = os.walk(self.input_dir)
        root, _, filenames = next(walk)
        print os.listdir('.')
        print self.output_dir
        if self.output_dir not in os.listdir('.'):
            os.mkdir('clean_data')
        for name in filenames:
            self.clean_data(os.path.join('raw_data', name), os.path.join('clean_data', name))

    def clean_data(self, input_filename, output_filename):
        with open(input_filename, 'rb') as input:
            with open(output_filename, 'wb') as out:
                reader = csv.reader(input)
                writer = csv.writer(out)
                writer.writerow(next(reader))
                for data_point in reader:
                    if len(data_point) < 5 or data_point[1] in EXCLUDE_DATA_SOURCE or data_point[2] in EXCLUDE_DATA:
                        continue
                    if len(data_point) > 5:
                        del data_point[4]
                    del data_point[3]
                    writer.writerow(data_point)


if __name__ == "__main__":
    data_cleaner = DataCleaner('raw_data', 'clean_data')
    data_cleaner.format_raw_data()
