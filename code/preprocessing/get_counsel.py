counsel_file = "E:/personal/Code/Python/LegalRetrieval/archive\\counsel_names.txt"
counsel_names = []
with open(counsel_file) as file:
    content = file.read()
    counsel_names = content.split(",")
    for counsel_name in counsel_names:
        if " and " in counsel_name:
            counsel_name_sp = counsel_name.split(" and ")
            counsel_name_sp = [counsel.strip() for counsel in counsel_name_sp]
            counsel_names.remove(counsel_name)
            counsel_names.extend(counsel_name_sp)
    counsel_names = [counsel.strip() for counsel in counsel_names]
    counsel_names = list(set(counsel_names))
    counsel_names = [counsel for counsel in counsel_names if len(counsel) > 0]
with open(counsel_file, "w") as file:
    file.write(",".join(counsel_names))