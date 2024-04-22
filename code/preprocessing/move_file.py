import shutil
import os

# read in out.txt
with open('E:/personal/Code/Python/LegalRetrieval/data\\file.txt', 'r') as file:
    data = file.readlines()
for file_name in os.listdir("E:/personal/Code/Python/LegalRetrieval/data/task1_train_files_2024"):
    if file_name+'\n' in data:
        shutil.move("E:/personal/Code/Python/LegalRetrieval/data/task1_train_files_2024/"+file_name, "E:/personal/Code/Python/LegalRetrieval/data/file_type_1/"+file_name)