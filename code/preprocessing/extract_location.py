from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.corpus import stopwords
import locationtagger
import os

def get_locations(text):
    # get location and store in type {location: Type}
    place_entity = locationtagger.find_locations(text = text)
    country_ = ""
    regions = []
    cities = []

    if len(place_entity.countries) < 1:
        country_ = "Canada"
    elif "Canada" in place_entity.countries:
        country_ = "Canada"
    else:
        country_ = place_entity.countries[0]
    
    for country in place_entity.country_regions:
        if country == country_:
            regions = place_entity.country_regions[country]
            break

    region_freq = {}
    for region in regions:
        region_freq[region] = text.count(region)
    region_ = max(region_freq, key=region_freq.get) if len(region_freq) > 0 else ""

    for region in place_entity.region_cities:
        if region == region_:
            cities = place_entity.region_cities[region]
            break
    
    city_freq = {}
    for city in cities:
        city_freq[city] = text.count(city)
    city_ = max(city_freq, key=city_freq.get) if len(city_freq) > 0 else ""

    return country_, region_, city_

data_folder = "E:/personal/Code/Python/LegalRetrieval/data/task1_train_files_2024"
output_file = "E:/personal/Code/Python/LegalRetrieval/code/temp_out\\location.txt"

for file_name in os.listdir(data_folder):
    with open(data_folder + "/" + file_name, 'r', encoding='utf-8') as file:
        text = file.read()
        country_, region_, city_ = get_locations(text)
        with open(output_file, 'a', encoding='utf-8') as output:
            output.write(file_name + "," + country_ + "," + region_ + "," + city_ + "\n")