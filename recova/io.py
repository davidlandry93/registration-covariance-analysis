import csv
import numpy as np
import sys

def read_xyz(instream):
    reader = csv.reader(instream, delimiter=' ')

    points = []
    for row in reader:
        points.append([float(row[0]), float(row[1]), float(row[2])])

    return np.array(points)

def convert_csv_to_xyz(instream, outstream):
    reader = csv.DictReader(instream)
    writer = csv.writer(outstream, delimiter=' ')

    for row in reader:
        writer.writerow([row['x'], row['y'], row['z']])

def csv2xyz_cli():
    convert_csv_to_xyz(sys.stdin, sys.stdout)

if __name__ == '__main__':
    cli()
