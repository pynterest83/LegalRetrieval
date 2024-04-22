import json
import os


# form of 1 case
# [
# {id: ,
#  order: ,
#  paragraph: ,
#  results: [
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

    for query_paragraph in case:
        candidate_case_set = set()
        case_score_list_dict = {}
        # list of dict, each represents the relevance score between qc and candidate case
        for paras in query_paragraph["results"]:
            if paras["id"] not in candidate_case_set:
                candidate_case_set.add(paras["id"])
                case_score_list_dict[paras["id"]] = {"query_id": query_paragraph["id"],
                                                     "query_order": query_paragraph["order"],
                                             "candidate_id": paras["id"],
                                             "candidate_order": paras["order"],
                                             "score": paras["score"]}
            else:
                #update score
                if case_score_list_dict[paras["id"]]["score"] < paras["score"]:
                    case_score_list_dict[paras["id"]] = paras
        # move item in case_score_list_dict to a list
        case_score_per_para = []
        for key in case_score_list_dict:
            case_score_per_para.append(case_score_list_dict[key])

        case_score_per_para = sorted(case_score_per_para, key=lambda x: x["score"], reverse=True)[:200]
        case_score_per_para = normalize(case_score_per_para)
        full_case_score.extend(case_score_per_para)
    return full_case_score


def normalize(case_score_per_para):
    for candidate_case in case_score_per_para:
        max_score = case_score_per_para[0]["score"]
        min_score = case_score_per_para[-1]["score"]
        candidate_case["score"] = (candidate_case["score"] - min_score) / (
                max_score - min_score + 1e-6)
    return case_score_per_para


def match_case(full_case_score):
    list_case_match = []
    top_cases_set = set()
    case_score_list_dict = {}
    for case_score_dict in full_case_score:
        if case_score_dict['candidate_id'] in top_cases_set:
            #update score
            if case_score_list_dict[case_score_dict['candidate_id']]["score"] < case_score_dict["score"]:
                case_score_list_dict[case_score_dict['candidate_id']] = case_score_dict
        else:
            top_cases_set.add(case_score_dict['candidate_id'])
            case_score_list_dict[case_score_dict['candidate_id']] = case_score_dict
    #move to list
    for key in case_score_list_dict:
        list_case_match.append(case_score_list_dict[key])
    # get the top 200 max score
    list_case_match = sorted(list_case_match, key=lambda x: x["score"], reverse=True)[:200]
    return list_case_match


if __name__ == "__main__":
    data_folder = "out"
    out_folder = "final_out"
    for file in os.listdir(data_folder):
        with open(os.path.join(data_folder, file), "r", encoding="utf-8") as f:
            case = json.load(f)
            full_case_score = match_para(case)
            list_case_match = match_case(full_case_score)
            with open(os.path.join(out_folder, file), "w", encoding="utf-8") as _file:
                json.dump(list_case_match, _file, indent=4)
