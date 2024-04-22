import json
import os
# form of 1 case
# [
# {id: , 
#  order: , 
#  paragraph: , 
#  result: [
#      {score: , 
#       id: , 
#       order: , 
#       content: 
#       }, 
#       ...]
#       }, 
# ...
# ]

def match_para(case):
    full_case_score = []
    for paragraph in case:
        case_score_per_para = []
        for result in paragraph["results"]:
            candidate_case = {}
            # if candidate case is not in the list
            if not any(candidate_case["id"] == result["id"] for candidate_case in case_score_per_para):
                candidate_case["id"] = result["id"]
                candidate_case["score"] = result["score"]
                case_score_per_para.append(candidate_case)
            else:
                for candidate_case in case_score_per_para:
                    if candidate_case["id"] == result["id"]:
                        candidate_case["score"] = max(candidate_case["score"], result["score"])
                        break
        case_score_per_para = sorted(case_score_per_para, key=lambda x: x["score"], reverse=True)[:200]
        full_case_score.append(case_score_per_para)
    return full_case_score

def normalize(case_score_per_para):
    for candidate_case in case_score_per_para:
        candidate_case["score"] = (candidate_case["score"] - case_score_per_para[-1]["score"]) / (case_score_per_para[0]["score"] - case_score_per_para[-1]["score"])
    return case_score_per_para

def match_case(full_case_score):
    list_case_match = []
    for case_score_per_para in full_case_score:
        # normalize the score
        case_score_per_para = normalize(case_score_per_para)
    # get the top 200 max score from all paragraph
    # set of unique case id with max score among all paragraph
    for case_score_per_para in full_case_score:
        for candidate_case in case_score_per_para:
            if not any(candidate_case["id"] == case["id"] for case in list_case_match):
                list_case_match.append(candidate_case)
            else:
                for case in list_case_match:
                    if case["id"] == candidate_case["id"]:
                        case["score"] = max(case["score"], candidate_case["score"])
                        break
    # get the top 200 max score
    list_case_match = sorted(list_case_match, key=lambda x: x["score"], reverse=True)[:200]
    return list_case_match

data_folder = ""
out_folder = ""
for file in os.listdir(data_folder):
        with open(os.path.join(data_folder, file), "r", encoding="utf-8") as f:
            case = json.load(f)
            full_case_score = match_para(case)
            list_case_match = match_case(full_case_score)
            with open(os.path.join(out_folder, file), "w", encoding="utf-8") as f:
                json.dump(list_case_match, f, indent=4)