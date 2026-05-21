"""
Fix the off-by-one bug in the code_representation field of rectangle JSON datasets.

generate_code_rectangle() used pts[3] (inclusive corner) as an exclusive range
endpoint, making every obstacle block one row and one column too small.
This script regenerates the correct code_representation for all rectangle samples.
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from representations import generate_code_rectangle

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

FILES = [
    os.path.join(DATA_DIR, 'rectangle_data_sg_iid.json'),
    os.path.join(DATA_DIR, 'rectangle_data_sg_ood.json'),
]

for path in FILES:
    data = json.load(open(path))
    changed = 0
    for s in data:
        x, y = len(s['world']), len(s['world'][0])
        new_repr = generate_code_rectangle(x, y, s['obstacles'], s['goals'], s['start'], s['points'])
        if new_repr != s['code_representation']:
            s['code_representation'] = new_repr
            changed += 1
    with open(path, 'w') as f:
        json.dump(data, f)
    print(f"{os.path.basename(path)}: fixed {changed}/{len(data)} samples")
