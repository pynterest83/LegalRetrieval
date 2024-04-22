import parsing
import os
import json

data_folder = "E:/personal/Code/Python/LegalRetrieval/data/task1_train_files_2024"
json_folder_1 = "E:/personal/Code/Python/LegalRetrieval/code/temp_out/extracted_train_1"
json_folder_2 = "E:/personal/Code/Python/LegalRetrieval/code/temp_out/extracted_train_2_2"
location = "E:/personal/Code/Python/LegalRetrieval/code/temp_out/location.txt"
out_folder = "E:/personal/Code/Python/LegalRetrieval/data/out"
raw_out_folder = "E:/personal/Code/Python/LegalRetrieval/data/raw_out"
longformer_out_folder = "E:/personal/Code/Python/LegalRetrieval/data/longformer_out"

location_dict = {}
# read line from location file
for line in open(location, "r", encoding="utf-8"):
    tokens = line.split(",")
    tokens[3] = tokens[3].replace("\n", "")
    if tokens[2] == "":
        tokens[3] = ""
    location_dict[tokens[0]] = tokens[1] + ", " + tokens[2] + ", " + tokens[3]

def extract_judge_name(paragraph):
    # seperate by " "
    # get the word before "J."
    token = paragraph.split(" ")
    for i in range(len(token)):
        if token[i] == "J.":
            return token[i-1][0:-1]
            
    return "Not Found"

def get_all_2(file_name):
    with open(json_folder_2 + "/" + file_name, "r", encoding="utf-8") as f:
        data = json.load(f)
        id = file_name.replace(".json", ".txt")
        counsels = data["counsels"]
        solicitors = data["solicitors"]
        judges = data["judge"]
        citation = data["citation"]
        bycitation = data["bycitation"]
        for judge in judges:
            judge['location'] = location_dict[id]
            judge['name'] = extract_judge_name(judge['name'])
        return id, counsels, solicitors, judges, citation, bycitation

def get_all_1(file_name):
    with open(json_folder_1 + "/" + file_name, "r", encoding="utf-8") as f:
        data = json.load(f)
        id = file_name.replace(".json", ".txt")
        counsels = data["counsels"]
        solicitors = data["solicitors"]
        judges = data["judge"]
        citation = data["citation"]
        bycitation = data["bycitation"]
        for judge in judges:
            judge['location'] = location_dict[id]
        return id, counsels, solicitors, judges, citation, bycitation

def get_paragraph(file_name):
    paragraphs = parsing.parsing(data_folder, file_name)
    return paragraphs

def get_raw_paragraph(file_name):
    paragraphs = parsing.raw_parse(data_folder, file_name)
    return paragraphs

def get_longformer_paragraph(file_name, raw_paragraphs):
    paragraphs, meta = parsing.longformer_parse(data_folder, file_name, raw_paragraphs)
    return paragraphs, meta

# for file_name in os.listdir(json_folder_2):
#     id, counsels, solicitors, judges, citation, bycitation = get_all_2(file_name)
#     paragraphs = get_paragraph(id)
#     data_out = {'id': id, 'counsels': counsels, 'solicitors': solicitors, 'judge': judges, 'citation': citation, 'bycitation': bycitation, 'paragraphs': paragraphs}
#     with open(out_folder + "/" + file_name, "w", encoding="utf-8") as file_out:
#         json.dump(data_out, file_out, ensure_ascii=False, indent=4)
#     file_out.close()

# for file_name in os.listdir(json_folder_1):
#     id, counsels, solicitors, judges, citation, bycitation = get_all_1(file_name)
#     paragraphs = get_paragraph(id)
#     data_out = {'id': id, 'counsels': counsels, 'solicitors': solicitors, 'judge': judges, 'citation': citation, 'bycitation': bycitation, 'paragraphs': paragraphs}
#     with open(out_folder + "/" + file_name, "w", encoding="utf-8") as file_out:
#         json.dump(data_out, file_out, ensure_ascii=False, indent=4)
#     file_out.close()

# for file_name in os.listdir(data_folder):
#     paragraphs = get_raw_paragraph(file_name)
#     data_out = {'paragraphs': paragraphs}
#     with open(raw_out_folder + "/" + file_name.replace(".txt", ".json"), "w", encoding="utf-8") as file_out:
#         json.dump(data_out, file_out, ensure_ascii=False, indent=4)
#     file_out.close()

for file_name in os.listdir(raw_out_folder):
    with open(raw_out_folder + "/" + file_name, "r", encoding="utf-8") as f:
        data = json.load(f)
        pre_paragraphs = data["paragraphs"]
        paragraphs, meta = get_longformer_paragraph(file_name, pre_paragraphs) 
        data_out = {'meta': meta,'paragraphs': paragraphs}
        with open(longformer_out_folder + "/" + file_name, "w", encoding="utf-8") as file_out:
            json.dump(data_out, file_out, ensure_ascii=False, indent=4)
        file_out.close()
