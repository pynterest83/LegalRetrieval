import json
import os
def open_json(file_path: str):
    content = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = json.load(file)
    except Exception as e:
        print(f'Error: {e}')
    return content
cite_times = {}
json_file = "E:/personal/Code/Python/LegalRetrieval/archive/task1_train_labels_2024.json"
json_content = open_json(json_file)
data_folder = "E:/personal/Code/Python/LegalRetrieval/archive/task1_train_files_2024"
for file_name in os.listdir(data_folder):
    if file_name in json_content.keys():
        for citation in json_content[file_name]:
            if citation not in cite_times.keys():
                cite_times[citation] = 1
            else:
                cite_times[citation] += 1
# sort the dictionary by value in descending order
sorted_cite_times = sorted(cite_times.items(), key=lambda x: x[1], reverse=True)
print(sorted_cite_times[:50])