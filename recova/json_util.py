#!/usr/bin/env python3

import argparse
import json
import sys

def json_cat_cli():
    parser = argparse.ArgumentParser(description='Merge json documents into a json list.')
    parser.add_argument('inputs', type=str, nargs='+', help='The files to concatenate')
    args = parser.parse_args()

    json_documents = []
    for f in args.inputs:
        with open(f) as jsonfile:
            json_document = json.load(jsonfile)
            json_documents.append(json_document)


    json.dump(json_documents, sys.stdout)
