# import the re module for regular expressions
import re, json, os, logging
import glob
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from nltk.corpus import stopwords
import locationtagger

def get_locations(text):
    # get location and store in type {location: Type}
    place_entity = locationtagger.find_locations(text = text)
    country_ = ""
    region_ = ""
    city_ = ""
    if len(place_entity.countries) < 1:
        country_ = "Canada"
    elif "Canada" in place_entity.countries:
        country_ = "Canada"
    else:
        country_ = place_entity.countries[0]
    
    for country in place_entity.country_regions:
        if country == country_:
            region_ = place_entity.country_regions[country][0]
            break
    
    for region in place_entity.region_cities:
        if region == region_:
            city_ = place_entity.region_cities[region][0]
            break

    return city_ + ", " + region_ + ", " + country_
def get_names(text):
    # get name and store in type {name: Type}
    names = {}
    nltk_results = ne_chunk(pos_tag(word_tokenize(text)))
    for nltk_result in nltk_results:
        if type(nltk_result) == Tree:
            name = ''
            for nltk_result_leaf in nltk_result.leaves():
                name += nltk_result_leaf[0] + ' '
            names[name] = nltk_result.label()
    return names

def get_paragraphs(text):
    pattern = r'(\[\d{1,4}\])'
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
            if check_no <= 6 and check_no >= 0:
                paragraphs.append(text[start:end])
                rolling_count = number + 1
            elif len(paragraphs) > 0:
                paragraphs[-1] += text[start:end]

    else:
        # If no matches, return the entire text
        paragraphs.append(text)

    return paragraphs

def extract_judge_name(paragraph):
    # seperate by " "
    # get the word before "J."
    token = paragraph.split(" ")
    for i in range(len(token)):
        if token[i] == "J.":
            return token[i-1][0:-1]
            
    return "Not Found"

def get_applicant(names, paragraph):
    for name in names:
        if names[name] == 'PERSON':
            # find the position of the name
            position = paragraph.find(name.strip())
            left_bound = position - 20 if position - 20 > 0 else 0
            right_bound = position + len(name) + 20 if position + len(name) + 20 < len(paragraph) else len(paragraph)
            # find the position of the word "applicant" ignore case
            applicant_position = paragraph.find("applicant", left_bound, right_bound)
            # find the position of the word plaintiff ignore case
            plaintiff_position = paragraph.find("plaintiff", left_bound, position + right_bound)
            if applicant_position != -1 or plaintiff_position != -1:
                return name.strip()
    return "Not found"

def get_defendant(names, paragraph):
    for name in names:
        if names[name] == 'PERSON':
            # find the position of the name
            position = paragraph.find(name.strip())
            left_bound = position - 20 if position - 20 > 0 else 0
            right_bound = position + len(name) + 20 if position + len(name) + 20 < len(paragraph) else len(paragraph)
            # find the position of the word "defendant" ignore case
            defendant_position = paragraph.find("defendant", left_bound, right_bound)
            # find the position of the word "respondent" ignore case
            respondent_position = paragraph.find("respondent", left_bound, right_bound)
            if defendant_position != -1 or respondent_position != -1:
                return name.strip()
    return "Not found"

def find_role(paragraph, position, applicant, defendant, name):
    left_bound = position - 100 if position - 100 > 0 else 0
    right_bound = position + len(name) + 100 if position + len(name) + 100 < len(paragraph) else len(paragraph)
    defendant_position = paragraph.find("defendant", left_bound, right_bound)
    if defendant_position == -1:
        defendant_position = paragraph.find(defendant)
    if defendant_position != -1:
        return "for the defendant"
    
    respondent_position = paragraph.find("respondent", left_bound, right_bound)
    if respondent_position == -1:
        respondent_position = paragraph.find(defendant)
    if respondent_position != -1:
        return "for the respondent"

    applicant_position = paragraph.find("applicant", left_bound, right_bound)
    if applicant_position == -1:
        applicant_position = paragraph.find(applicant)
    if applicant_position != -1:
        return "for the applicant"
    
    plaintiff_position = paragraph.find("plaintiff", left_bound, right_bound)
    if plaintiff_position == -1:
        plaintiff_position = paragraph.find(applicant)
    if plaintiff_position != -1:
        return "for the plaintiff"

def get_counsel(paragraph, applicant, defendant, counsels, solicitors):
    names = get_names(paragraph)
    role = ""
    for name in names:
        if names[name] == 'PERSON' and name.strip() != "Failure" and name.strip() not in applicant and name.strip() not in defendant :
            # find the position of the name
            position = paragraph.find(name.strip())
            left_bound = position - 50 if position - 50 > 0 else 0
            right_bound = position + len(name) + 50 if position + len(name) + 50 < len(paragraph) else len(paragraph)
            role = find_role(paragraph, position, applicant, defendant, name)
            
            # find the position of the word "counsel" ignore case
            counsel_position = paragraph.find("counsel", left_bound, right_bound)
            if counsel_position != -1 and role != None:
                counsels.append({"name": name.strip(), "role": role})
                continue
            # find the position of the word "solicitor" ignore case
            solicitor_position = paragraph.find("solicitor", left_bound, right_bound)
            if solicitor_position != -1 and role != None:
                solicitors.append({"name": name.strip(), "role": role})
                continue

def get_organization(names):
    organizations = "Federal Court of Canada"
    for name in names:
        if names[name] == 'ORGANIZATION' and "Court" in name and len(name.split(" ")) == 3:
            organizations = name
    return organizations

def open_json(file_path: str):
    content = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = json.load(file)
    except Exception as e:
        print(f'Error: {e}')
    return content

def extract_bycitation(content, file_name):
    bycitation = []
    for key, value in content.items():
        if file_name in value:
            bycitation.append(key)
    return bycitation

def extract_citation(content, file_name):
    citation = []
    if file_name in content.keys():
        citation = content[file_name]
    return citation

def extract_editor(text):
    matches = re.findall(pattern=r'.*Editor.*', string=text)
    if len(matches) == 0:
        return ""
    # get from the second word to the end
    return ' '.join(matches[0].split(" ")[1:])

def find_same_counsel_name(names):
    for name in names:
        if names[name] == 'PERSON':
            for counsel_name in counsel_names:
                if name.strip() in counsel_name:
                    return counsel_name
    return "Not found"

def extract_data(text, file_name):
    metadata = {}
    all_names = get_names(text)
    paragraphs = get_paragraphs(text)
    applicant = ""
    defendant = ""
    counsels = []
    solicitors = []
    for paragraph in paragraphs:
        names = get_names(paragraph)
        applicant = get_applicant(names, paragraph)
        defendant = get_defendant(names, paragraph)
        if applicant != "Not found":
            break
        if defendant != "Not found":
            break
    for paragraph in paragraphs:
        get_counsel(paragraph, applicant, defendant, counsels, solicitors)
    
    metadata["id"] = file_name
    if len(counsels) > 0: 
        metadata["counsels"] = counsels
    else: metadata["counsels"] = [{"name": find_same_counsel_name(all_names), "role": "Not found"}]
    if len(solicitors) > 0: 
        metadata["solicitors"] = solicitors
    else: metadata["solicitors"] = [{"name": "Not found", "role": "Not found"}]
    if metadata["counsels"][0]["name"] == "Not found" and metadata["solicitors"][0]["name"] == "Not found":
        return metadata
    citations = extract_citation(json_content, file_name)
    bycitations = extract_bycitation(json_content, file_name)
    judge_name = extract_judge_name(paragraphs[0])
    locations = get_locations(text)
    editor = extract_editor(text)
    organizations = get_organization(all_names)
    judge = [{"location":locations, "name": judge_name, "organization": organizations}]
    metadata["judge"] = judge
    metadata["citation"] = citations
    metadata["bycitation"] = bycitations
    metadata["editor"] = editor

    return metadata

data_folder = "E:/personal/Code/Python/LegalRetrieval/archive/unextracted_data"
json_file = "E:/personal/Code/Python/LegalRetrieval/archive/task1_train_labels_2024.json"
json_content = open_json(json_file)
counsel_file = "E:/personal/Code/Python/LegalRetrieval/archive\\counsel_names.txt"
counsel_names = []
with open(counsel_file) as file:
    content = file.read()
    counsel_names = content.split(",")
    counsel_names = [name.strip() for name in counsel_names]
for file_name in os.listdir(data_folder):
    file_path = os.path.join(data_folder, file_name)
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        metadata = extract_data(text, file_name)
        if metadata["counsels"][0]["name"] == "Not found" and metadata["solicitors"][0]["name"] == "Not found":
            continue
        file_name = file_name.split(".")[0]
        with open("E:/personal/Code/Python/LegalRetrieval/archive/extracted_data2/" + file_name + ".json", 'w', encoding='utf-8') as file_out:
            json.dump(metadata, file_out, ensure_ascii=False, indent=4)
        file_out.close()