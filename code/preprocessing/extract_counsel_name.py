def get_name(line):
    name = line.split(',')[0]
    return name.strip()

with open("E:/personal/Code/Python/LegalRetrieval/archive\\counsel_lines.txt") as file:
    lines = file.readlines()
    lines = [line for line in lines if line.strip()]
    names = [get_name(line) for line in lines]
    # remove duplicates
    names = list(set(names))
    # write to file
    with open("E:/personal/Code/Python/LegalRetrieval/archive\\counsel_names.txt", "w") as file:
        for name in names:
            file.write(name + ",")