import os
import json
"""
category | living room max appearances | kitchen max appearances|  dining room max appearances

"""

object_count = {}

# path_to_dictionaries = os.getcwd() + "/object_plane_dictionaries"
path_to_dictionaries = os.getcwd() + "/default"

col = 0

for dictionary_name in os.listdir(path_to_dictionaries):
    json_file = open(path_to_dictionaries + f"/{dictionary_name}")
    op_dict = json.load(json_file)
    
    dict_type = dictionary_name.split("_")[0]
    if dict_type == "living": col = 0
    if dict_type == "kitchen": col = 1
    if dict_type == "dining": col = 2
    
    for obj_name in list(op_dict.keys()):
        
        category_name = obj_name.split("_")[1]
        second_word = obj_name.split("_")[2]
        if category_name == "cell": category_name = "cell phone"
        if second_word == "bat": category_name = "baseball bat"
        if second_word == "glove": category_name = "baseball glove"
        if category_name == "potted": category_name = "potted plant"
        if category_name == "tennis": category_name = "tennis racket"
        if category_name == "hair": category_name = "hair drier"
        if category_name == "ball": category_name = "sports ball"
        
        if category_name in list(object_count.keys()):
            object_count[category_name][dict_type] += op_dict[obj_name]["max_number"]
        else:
            object_count[category_name] = {}
            object_count[category_name]["living"] = 0
            object_count[category_name]["kitchen"] = 0
            object_count[category_name]["dining"] = 0
            object_count[category_name][dict_type] += op_dict[obj_name]["max_number"]
            
print("\nNumber of Categories Used")
print(len(object_count.keys()))
with open("analyze_object_plane_dictionaries.json", 'w') as f:
    json.dump(object_count, f, indent=4)