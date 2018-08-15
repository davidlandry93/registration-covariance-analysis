#!/usr/bin/env python3

import argparse
import subprocess


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('begin', type=int)
    parser.add_argument('end', type=int)
    parser.add_argument('batch_size', type=int)
    parser.add_argument('--dry-run', '-d', action='store_true')
    args = parser.parse_args()

    steps = list(range(args.begin, args.end, args.batch_size))
    if steps[-1] != args.end:
        steps.append(args.end)

    for i in range(len(steps) - 1):
        beg = steps[i]
        end = steps[i+1]

        cmd = 'export BEGIN={}; export END={}; sbatch batches_of_pairs_cluster.sh'.format(beg, end)
        print(cmd)

        if not args.dry_run:
            subprocess.run(cmd, shell=True)
    
