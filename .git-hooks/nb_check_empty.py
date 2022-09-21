import json
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser(description='Script to check that the given Jupyter notebook has no output cells.')
parser.add_argument('file', type=Path)
args = parser.parse_args()

if not args.file.is_file():
    print(f'File not found: {args.file}')
    exit(0)

if args.file.suffix != '.ipynb':
    print(f'File must be a Jupyter notebook. Got {args.file}')
    exit(0)

with open(args.file) as f:
    data = json.load(f)

for cell in data['cells']:
    if cell.get('outputs', None):
        exit(1)

exit(0)
