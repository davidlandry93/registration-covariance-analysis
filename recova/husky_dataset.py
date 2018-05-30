
import csv
from pathlib import Path
import numpy as np

from recov.pointcloud_io import points_of_pcd
from recova.util import quat_to_rot_matrix

class HuskyDataset:
    def __init__(self, path):
        self.path = Path(path)

        self.map_pose, self.map_times, self.map_files = self.import_csv_index(self.path / 'maps.csv')
        self.scan_pose, self.scan_times, self.scan_files = self.import_csv_index(self.path / 'scans.csv')


    def import_csv_index(self, path):
        index = []
        with path.open() as f:
            reader = csv.DictReader(f)
            map_entries = [row for row in reader]

        poses = np.empty((len(map_entries), 4, 4))
        times = np.empty(len(map_entries))
        files = []

        for i, entry in enumerate(map_entries):
            poses[i] = np.identity(4)
            poses[i, 0:3, 3] = (entry['tx'], entry['ty'], entry['tz'])

            poses[i, 0:3, 0:3] = quat_to_rot_matrix((float(entry['qx']),
                                                     float(entry['qy']),
                                                     float(entry['qz']),
                                                     float(entry['qw'])))
            times[i] = float(entry['timestamp'])
            files.append(entry['filename'])

        return poses, times, files

    def points_of_pcd_file(self, pcd):
        return points_of_pcd(pcd)

    def points_of_scan(self, scan_id):
        return self.points_of_pcd_file(self.path / self.scan_files[scan_id])

    def points_of_map(self, map_id):
        return self.points_of_pcd_file(self.path / self.map_files[map_id])

    def points_of_reading(self, reading):
        return self.points_of_scan(reading)

    def points_of_reference(self, reference):
        return self.points_of_map(reference)

    def find_pair(self, map_id, scan_delay):
        """
        For a given map_id, find a scan id that was taken scan_delay seconds after the map was published.
        """
        map_time = self.map_times[map_id]

        scan_id = np.argmax(self.scan_times > map_time - scan_delay)

        return (scan_id, map_id)
