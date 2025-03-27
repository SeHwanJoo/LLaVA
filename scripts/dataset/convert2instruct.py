import json
import csv
from pathlib import Path


json_data = json.load(open("something-something-v2-id2label.json", "r"))
splits = ["train", "val", "test"]
id = 1
for split in splits:
    results = []
    csv_file_path = f"{split}.csv"
    csv_reader = csv.reader(open(csv_file_path, "r"))
    for row in csv_reader:
        file_path, label = row[0].split(" ")
        file_name = str(Path(file_path).name)
        answer = json_data[label]
        data = {
            "id": str(id),
            "video": file_name,
            "conversations": [
                {
                    "from": "human",
                    "value": "Describe this video.\n<video>"
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ]
        }
        results.append(data)
        id += 1
    json.dump(
        results,
        open(f"something-something-v2_{split}.json", "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2
    )




