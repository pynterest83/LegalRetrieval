import re
from langdetect import detect

header_marks = [["I\.", "A\.", "\(a\)"], ["II\.", "B\.", "\(b\)"], ["III\.", "C\.", "\(c\)"], ["IV\.", "D\.", "\(d\)"], ["V\.", "E\.", "\(e\)"], ["VI\.", "F\.", "\(f\)"], ["VII\.", "G\.", "\(g\)"], ["VIII\.", "H\.", "\(h\)"], ["IX\.", "I\.", "\(i\)"], ["X\.", "J\.", "\(j\)"], ["XI\.", "K\.", "\(k\)"], ["XII\.", "L\.", "\(l\)"], ["XIII\.", "M\.", "\(m\)"], ["XIV\.", "N\.", "\(n\)"], ["XV\.", "O\.", "\(o\)"], ["XVI\.", "P\.", "\(p\)"], ["XVII\.", "Q\.", "\(q\)"], ["XVIII\.", "R\.", "\(r\)"], ["XIX\.", "S\.", "\(s\)"], ["XX\.", "T\.", "\(t\)"], ["XXI\.", "U\.", "\(u\)"], ["XXII\.", "V\.", "\(v\)"], ["XXIII\.", "W\.", "\(w\)"], ["XXIV\.", "X\.", "\(x\)"], ["XXV\.", "Y\.", "\(y\)"], ["XXVI\.", "Z\.", "\(z\)"]]

catch_list_1 = ["SUMMARY", "Summary", "Summaries", "SUMMARIES"]
catch_list_2 = ["BACKGROUND", "Background", "Backgrounds", "BACKGROUNDS"]
catch_list_3 = ["DECISION", "Decision", "Decisions", "DECISIONS"]
catch_list_4 = ["REASONING", "Reasoning", "Reasons", "REASONS", "Reason", "REASON"]
catch_list_5 = ["ISSUE", "Issue", "Issues", "ISSUES"]
catch_list_6 = ["JUDGMENT", "Judgment"]
catch_list_7 = ["CONCLUSION", "Conclusion", "Conclusions", "CONCLUSIONS"]
catch_list_8 = ["ANALYSIS", "Analysis"]
catch_list_9 = ["INTRODUCTION", "Introduction"]
catch_list_10 = ["LEGISLATIVE FRAMEWORK", "Legislative Framework"]
catch_list_11 = ["STANDARD OF REVIEW", "Standard of Review", "Standard of review"]
catch_list_12 = ["OVERVIEW", "OVERVIEWS", "Overview", "Overviews"]

header_names = [catch_list_1, catch_list_2, catch_list_3, catch_list_4, catch_list_5, catch_list_6, catch_list_7, catch_list_8, catch_list_9, catch_list_10, catch_list_11, catch_list_12]

catch_next = ["[", "  ", " ", ":", "\n", "•", "• ", "."]

suppressed_list = ["FRAGMENT_SUPPRESSED", "REFERENCE_SUPPRESSED", "CITATION_SUPPRESSED", "DATE_SUPPRESSED"]

def filter_1(text):
    range_list = []
    for i in range(26):
        for header in header_marks[i]:
            # find all the indices of the header
            indices = [m.start() for m in re.finditer(header, text)]
            for index in indices:
                for i in range(len(header_names)):
                    for name in header_names[i]:
                        if name in text[index:index+30]:
                            range_list.append(index)
                            break
    return range_list

def filter_2(text):
    range_list = []
    for i in range(len(header_names)):
        for name in header_names[i]:
            if text.find(name) != -1:
                if text[text.find(name)+len(name)] in catch_next:
                    range_list.append(text.find(name))
                    break
    return range_list

def get_header_range(text):
    range_list = filter_1(text)
    if len(range_list) == 0:
        range_list = filter_2(text)
    return range_list

def get_paragraphs(text):
    pattern = r'(\[\d{1,9}\])'
    paragraphs = []
    text = text.replace('\n', ' ').replace('•',' ')
    text = re.sub(r'\s+', ' ', text)

    # Use regex to split the raw text into paragraphs based on [1], [2] ... formatting:

    start_indices = [match.start() for match in re.finditer(pattern, text)]

    if start_indices:
        rolling_count = 1

        for i in range(len(start_indices)):
            start = start_indices[i]
            target_text = text[start:start+6]
            match = re.search(pattern, target_text)
            if match:
                number = int(match.group().strip('[]').strip('.'))
            else:
                continue

            end = start_indices[i + 1] if i + 1 < len(start_indices) else len(text)

            # check if the bracketed number is within buffer range of the previous number:
            check_no = number - rolling_count
            if check_no <= 200 and check_no >= 0:
                paragraphs.append(text[start:end])
                rolling_count = number + 1
            elif len(paragraphs) > 0:
                paragraphs[-1] += text[start:end]

    else:
        # If no matches, return the entire text
        paragraphs.append(text)

    return paragraphs

def get_meta(text):
    # get the text from start to [1]
    start_index = text.find("[1]")
    return text[:start_index]

def remove_index_paragraph(paragragh: str):
    
    paragragh = paragragh.replace('?','.')
    paragragh = paragragh.replace('!','.')
    paragragh = paragragh.replace('。','.')


    paragragh = paragragh.split('.')

    index = []

    while len(paragragh[-1].split(' ')) <= 5:
        index.append(paragragh[-1])
        if len(paragragh) == 1: 
            return '.'.join(paragragh)
        paragragh.pop()
    
    while len(paragragh[-1].split(' ')) <= 3:
        index.append(paragragh[-1])
        if len(paragragh) == 1: 
            return '.'.join(paragragh)
        paragragh.pop()

    return '.'.join(paragragh)  

def clean_text(text):
    numbering_pattern = r'\[\d+\]'
    angle_pattern = r'[<>]'
    end_pattern = r"\[End of document\]"
    hidden_parentheses = r'[(*)]'
    double_space = r'\s{2,}'

    text = text.replace('\n', ' ').replace('•',' ')
    text = re.sub(numbering_pattern,' ', text)
    text = re.sub(angle_pattern,' ', text)
    text = re.sub(end_pattern,' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(hidden_parentheses, '.', text)
    text = re.sub(double_space, '', text)

    text = text.strip()

    return text

def clean_suppressed(paragragh: str):
    for suppressed in suppressed_list:
        paragragh = paragragh.replace(suppressed, "")
    return paragragh

def get_header(text):
    for i in range(len(header_names)):
        for name in header_names[i]:
            if name in text[0:30]:
                return name
    return ""

def remove_non_eng(paragraph):
    try:
        if detect(paragraph) != 'en':
            return ""
    except:
        return paragraph
    return paragraph

def parsing(data_folder, file_name):
    paragraphs = []
    with open(data_folder + "/" + file_name, 'r', encoding='utf-8') as file:
        text = file.read()
    range_list = get_header_range(text)
    range_list = list(set(range_list))
    range_list.insert(0, 0)
    range_list.sort()
    for i in range(len(range_list)):
        if i == len(range_list) - 1:
            sub_text = text[range_list[i]:]
        else:
            sub_text = text[range_list[i]:range_list[i+1]]
        if len(sub_text) < 100: continue
        raw_paragraphs = get_paragraphs(sub_text)
        header = get_header(sub_text)
        clean_paragraphs = []
        for paragraph in raw_paragraphs:
            if len(paragraph) < 100: continue
            paragraph = clean_suppressed(paragraph)
            paragraph = clean_text(paragraph)
            if len(paragraph) < 100: continue
            paragraph = remove_index_paragraph(paragraph)
            # paragraph = remove_non_eng(paragraph)
            clean_paragraphs.append(paragraph)
        if len(clean_paragraphs) == 0: continue
        paragraphs.append({'header': header, 'raw_paragraphs': raw_paragraphs, 'clean_paragraphs': clean_paragraphs})
    return paragraphs

def raw_parse(data_folder, file_name):
    with open(data_folder + "/" + file_name, 'r', encoding='utf-8') as file:
        text = file.read()
    paragraphs = get_paragraphs(text)
    for paragraph in paragraphs:
        paragraph = clean_text(paragraph)
        paragraph = remove_non_eng(paragraph)
        paragraph = remove_index_paragraph(paragraph)
        paragraph = remove_non_eng(paragraph)
    return paragraphs

def longformer_parse(data_folder, file_name, raw_paragraphs):
    # replace json and txt
    file_name = file_name.replace(".json", ".txt")
    with open(data_folder + "/" + file_name, 'r', encoding='utf-8') as file:
        text = file.read()
    meta = get_meta(text)
    meta = clean_suppressed(meta)
    meta = clean_text(meta)
    paragraphs = []
    for paragraph in raw_paragraphs:
        paragraph = clean_suppressed(paragraph)
        paragraph = clean_text(paragraph)
        paragraph = remove_non_eng(paragraph)
        paragraphs.append(paragraph)
    return paragraphs, meta

# data_folder = "E:/personal/Code/Python/LegalRetrieval/data/task1_train_files_2024"
# output_file = "E:/personal/Code/Python/LegalRetrieval/code/temp_out\\out.txt"

# for file_name in os.listdir(data_folder):
#     if file_name != "000469.txt": continue
#     with open(data_folder + "/" + file_name, 'r', encoding='utf-8') as file:
#         text = file.read()
#         range_list = get_header_range(text)
#         range_list = list(set(range_list))
#         range_list.insert(0, 0)
#         range_list.sort()
#         for i in range(len(range_list)):
#             if i == len(range_list) - 1:
#                 sub_text = text[range_list[i]:]
#             else:
#                 sub_text = text[range_list[i]:range_list[i+1]]
#             if len(sub_text) < 100: continue
#             paragraphs = get_paragraphs(sub_text)
#             with open(output_file, 'a', encoding='utf-8') as out_file:
#                 out_file.write(get_header(sub_text) + '\n')
#                 for paragraph in paragraphs:
#                     if len(paragraph) < 100: continue
#                     paragraph = remove_index_paragraph(paragraph)
#                     paragraph = clean_suppressed(paragraph)
#                     paragraph = clean_text(paragraph)
#                     out_file.write(paragraph + '\n')