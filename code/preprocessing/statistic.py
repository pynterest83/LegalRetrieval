# Luật sư X hay cite đến các case nào
# Kiểu luật sư X cite case Y 3 lần case Z 4 lần

import json
import os

lawyers_cite = []

folder_1 = "E:/personal/Code/Python/LegalRetrieval/archive/extracted_train_2"
folder_2 = "E:/personal/Code/Python/LegalRetrieval/archive/extracted_train_1"

for file_name in os.listdir(folder_1):
    with open(folder_1 + "/" + file_name, "r", encoding="utf-8") as f:
        data = json.load(f)
        counsels = data[1]['counsel']
        solicitors = data[2]['solicitors']
        cites = data[4]['citation']

        for counsel in counsels:
            counsel['name'] = counsel['name'].strip()
            if counsel['name'] == "":
                continue
            if counsel['name'][0].islower():
                continue
            name = counsel['name']
            if " not " in name or " Not " in name or " No " in name or "No " in name or " no " in name or "Not " in name or " Inc." in name:
                continue
            if "for the respondent" in name or "for the applicant" in name:
                # remove "for" in name
                name = name.replace("for the respondent", "")
                name = name.replace("for the applicant", "")
            lawyer_cite = []
            for lawyer in lawyers_cite:
                if lawyer['name'] == name:
                    lawyer_cite = lawyer['cite']
                    # remove current lawyer
                    lawyers_cite.remove(lawyer)
                    break
            for cite in cites:
                if cite not in [x['case'] for x in lawyer_cite]:
                    lawyer_cite.append({'case': cite, 'count': 1})
                else:
                    for lc in lawyer_cite:
                        if lc['case'] == cite:
                            lc['count'] += 1
            lawyers_cite.append({'name': name, 'role': 'counsel', 'cite': lawyer_cite})       
        
        for solicitor in solicitors:
            solicitor['name'] = solicitor['name'].strip()
            if solicitor['name'] == "":
                continue
            if solicitor['name'][0].islower():
                continue
            name = solicitor['name']
            tokens = name.split(",")
            name = tokens[0].strip()
            role = 'solicitor'
            if " not " in name or " Not " in name or " No " in name or "No " in name or " no " in name or "Not " in name:
                continue
            if "for the respondent" in name or "for the applicant" in name:
                # remove "for" in name
                name = name.replace("for the respondent", "")
                name = name.replace("for the applicant", "")
            lawyer_cite = []
            for lawyer in lawyers_cite:
                if lawyer['name'] == name:
                    lawyer_cite = lawyer['cite']
                    lawyers_cite.remove(lawyer)
                    break
            for cite in cites:
                if cite not in [x['case'] for x in lawyer_cite]:
                    lawyer_cite.append({'case': cite, 'count': 1})
                else:
                    for lc in lawyer_cite:
                        if lc['case'] == cite:
                            lc['count'] += 1
            lawyers_cite.append({'name': name, 'role': role, 'cite': lawyer_cite})

for file_name in os.listdir(folder_2):
    with open(folder_2 + "/" + file_name, "r", encoding="utf-8") as f:
        data = json.load(f)
        counsels = data["counsels"]
        solicitors = data["solicitors"]
        cites = data["citation"]

        for counsel in counsels:
            counsel['name'] = counsel['name'].strip()
            if counsel['name'] == "":
                continue
            if counsel['name'][0].islower():
                continue
            name = counsel['name']
            if " not " in name or " Not " in name or " No " in name or "No " in name or " no " in name or "Not " in name or " Inc." in name:
                continue
            if "for the respondent" in name or "for the applicant" in name:
                # remove "for" in name
                name = name.replace("for the respondent", "")
                name = name.replace("for the applicant", "")
            lawyer_cite = []
            for lawyer in lawyers_cite:
                if lawyer['name'] == name:
                    lawyer_cite = lawyer['cite']
                    # remove current lawyer
                    lawyers_cite.remove(lawyer)
                    break
            for cite in cites:
                if cite not in [x['case'] for x in lawyer_cite]:
                    lawyer_cite.append({'case': cite, 'count': 1})
                else:
                    for lc in lawyer_cite:
                        if lc['case'] == cite:
                            lc['count'] += 1
            lawyers_cite.append({'name': name, 'role': 'counsel', 'cite': lawyer_cite}) 

        for solicitor in solicitors:
            solicitor['name'] = solicitor['name'].strip()
            if solicitor['name'] == "":
                continue
            if solicitor['name'][0].islower():
                continue
            name = solicitor['name']
            tokens = name.split(",")
            name = tokens[0].strip()
            role = 'solicitor'
            if " not " in name or " Not " in name or " No " in name or "No " in name or " no " in name or "Not " in name:
                continue
            if "for the respondent" in name or "for the applicant" in name:
                # remove "for" in name
                name = name.replace("for the respondent", "")
                name = name.replace("for the applicant", "")
            lawyer_cite = []
            for lawyer in lawyers_cite:
                if lawyer['name'] == name:
                    lawyer_cite = lawyer['cite']
                    lawyers_cite.remove(lawyer)
                    break
            for cite in cites:
                if cite not in [x['case'] for x in lawyer_cite]:
                    lawyer_cite.append({'case': cite, 'count': 1})
                else:
                    for lc in lawyer_cite:
                        if lc['case'] == cite:
                            lc['count'] += 1
            lawyers_cite.append({'name': name, 'role': role, 'cite': lawyer_cite})   

count = {}

with open("E:/personal/Code/Python/LegalRetrieval/archive/statistic_result/stats_out.txt", "w", encoding="utf-8") as f:
    # not write in json
    for lawyer in lawyers_cite:
        if len(lawyer['cite']) == 0:
            continue
        # rank lawyer cite case by count
        lawyer['cite'] = sorted(lawyer['cite'], key=lambda x: x['count'], reverse=True)
        f.write(lawyer['name'] + " " + lawyer['role'] + " " + str(len(lawyer['cite'])) + "\n")
        for cite in lawyer['cite']:
            count[cite['case']] = count.get(cite['case'], 0) + cite['count']
            f.write(cite['case'] + " " + str(cite['count']) + "\n")
        f.write("\n")

# sort count
count = sorted(count.items(), key=lambda x: x[1], reverse=True)
# print top 50 case cite most
for c in count[:50]:
    print(c[0] + " " + str(c[1]))

