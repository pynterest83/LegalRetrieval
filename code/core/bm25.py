from elasticsearch import Elasticsearch
import os
import json

def login(account, password):
    client = Elasticsearch(hosts="http://localhost:9200", basic_auth=(account, password), verify_certs=False)
    return client

def mapping(client, _index):
    _mapping = {
        "settings": {
            "number_of_shards": 1,
            "index": {
                "similarity": {
                    "default": {
                        "type": "BM25",
                        "b": 0.5,
                        "k1": 1.5
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "paragraph": {
                    "type": "text"
                }
            }
        }
    }

    response = client.indices.create(
        index=_index,
        body=_mapping,
        ignore=400
    )

def indexing(client, _index, input_list):
    for i in range(len(input_list)):
        _id = input_list[i]["id"]
        _paragraphs = input_list[i]["paragraphs"]
        for _num in range(len(_paragraphs)):
            paradoc = {
                "id" : _id,
                "num" : _num + 1,
                "paragraph" : _paragraphs[_num]
            }

            client.index(index=_index, body=paradoc)
        print("Indexing case " + _id + " done!")

def searching(client, _index, _query:str):
    query = {
        "size": 500,
        "query": {
            "match": { "paragraph": _query }
        },
        "sort": [
            { "_score": "desc" }
        ]
    }

    return client.search(index=_index, body=query)

def get_result(response, query_id):
    results = []
    for hit in response['hits']['hits']:
        para_id = hit['_source']['id']
        if para_id == query_id:
            continue
        para_score = hit['_score']
        para_num = hit['_source']['num']
        para_content = hit['_source']['paragraph']
        results.append([para_score, para_id, para_num, para_content])
    return results

def get_case_query(case, _index, client):
    case_results = []
    _id = case["id"]
    _paragraphs = case["paragraphs"]
    for _num in range(len(_paragraphs)):
        response = searching(client, _index, _paragraphs[_num])
        results = get_result(response, _id) # a list
        para_results = []
        for result in results:
            query_results = {}
            query_results["score"] = result[0]
            query_results["id"] = result[1]
            query_results["order"] = result[2]
            query_results["content"] = result[3]
            para_results.append(query_results)
        case_results.append({"id": _id, "order": _num + 1, "paragraph": _paragraphs[_num], "results": para_results})
    return case_results

def get_output(input_list, _index, client):
    out_put_folder = "E:/personal/Code/Python/LegalRetrieval/code/temp_out/bm25_out"
    for case in input_list:
        case_results = get_case_query(case, _index, client)
        # write to json file
        json_file = case["id"] + ".json"
        with open(os.path.join(out_put_folder, json_file), "w", encoding="utf-8") as f:
            json.dump(case_results, f, ensure_ascii=False, indent=4)

account = "elastic"
password = "huyquang004"
client = login(account, password)
_index = "es_coliee_2024"
# client.indices.delete(index=_index, ignore=[400, 404])
if not client.indices.exists(index=_index):
    mapping(client, _index)
if client.indices.exists(index=_index):
    print("Index existed")
    data_folder = "E:/personal/Code/Python/LegalRetrieval/data/raw_out"
    input_list = []
    count = 0
    for file in os.listdir(data_folder):
        with open(os.path.join(data_folder, file), "r", encoding="utf-8") as f:
            data = json.load(f)
            file_id = file.split(".")[0]
            paragraphs = data["paragraphs"]
            input_list.append({"id": file_id, "paragraphs": paragraphs})
            print("Reading case " + file_id + " done!")
            count += 1
            if count == 100:
                break
    indexing(client, _index, input_list)
    get_output(input_list, _index, client)