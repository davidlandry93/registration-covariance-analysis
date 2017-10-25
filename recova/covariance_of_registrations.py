#!/usr/bin/python3 

import argparse
import json
import sys

from recova.registration_dataset import registrations_of_dataset

def cli():
    dataset = json.load(sys.stdin)
    registrations = registrations_of_dataset(dataset)

    print(registrations)

if __name__ == '__main__':
    cli()
