import json
from pathlib import Path
import csv
import os

# JSON File path
json_path = Path(__file__).resolve().parent / "keyframe_annotations.json"

with open(json_path, 'r') as f:
    data = json.load(f)

# CSV Save in the same directory
output_path = os.path.join(os.path.dirname(json_path), 'output.csv')

with open(output_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['key', 'time'])
    for key, value in data.items():
        writer.writerow([f'video/{key}', value['time']])  # Add the video/ prefix

print(f"CSV generated: {output_path}")
