import os
import json
from datetime import datetime
import copy

""" Merge datasets in specified directory
    assumes all datasets have the same "cataegories" key and aassociated field values
"""
path_to_datasets = os.getcwd() + "/datasets/"

instance_dictionaries_list = []
ref_dictionaries_list = []
for dataset in os.listdir(path_to_datasets):
    with open(path_to_datasets + dataset + "/instances.json") as json_file:
        instance_dictionaries_list.append(json.load(json_file))
    with open(path_to_datasets + dataset + "/refs.json") as json_file:
        ref_dictionaries_list.append(json.load(json_file))

# create the merged datset directory
current_time_stamp = str(datetime.now())
current_time_stamp = "merged_test"
if(os.path.exists(os.getcwd() + "/merged_datasets/" + current_time_stamp)):
    print(f"Merged Dataset {os.getcwd() + '/merged_datasets/' + current_time_stamp} already exists")
else:
    os.mkdir(os.getcwd() + "/merged_datasets/" + current_time_stamp)
    os.mkdir(os.getcwd() + "/merged_datasets/" + current_time_stamp + "/images")
    os.mkdir(os.getcwd() + "/merged_datasets/" + current_time_stamp + "/annotated")

### Create merged instnaces.json file

# initialize merged instances using the first file
merged_instances_dict = {}
merged_instances_dict["info"] = {
    "description": f"This is dev 0.1 version of the 2024 RefCOCO 3D Synthetic Spatial dataset. \
                    Merged datasets {os.listdir(path_to_datasets)}",
    "version": "0.1",
    "year": 2024,
    "contributor": "University of Michigan",
    "date_created": f"{current_time_stamp}"
}
merged_instances_dict["images"] = instance_dictionaries_list[0]["images"]
merged_instances_dict["annotations"] = instance_dictionaries_list[0]["annotations"]
merged_instances_dict["categories"] = instance_dictionaries_list[0]["categories"]

last_image_id = merged_instances_dict["images"][-1]


# increment instnaces.json - images
next_image_id = merged_instances_dict["images"][-1]["id"] + 1
old_to_new_image_id_mapping = {}
next_annotation_id = merged_instances_dict["annotations"][-1]["id"] + 1
old_to_new_annotation_id_mapping = {}
for instances_dict in instance_dictionaries_list[1:]:
    
    for image_dict in instances_dict["images"]:
        
        incremented_image_dict = copy.deepcopy(image_dict)
        old_to_new_image_id_mapping[incremented_image_dict["id"]] = next_image_id
        incremented_image_dict["file_name"] = str(next_image_id).zfill(6) + ".png"
        incremented_image_dict["id"] = next_image_id
        
        merged_instances_dict["images"].append(incremented_image_dict)
        
        next_image_id += 1
        
    for anno_dict in instances_dict["annotations"]:
        incremented_anno_dict = copy.deepcopy(anno_dict)
        old_to_new_annotation_id_mapping[incremented_anno_dict["id"]] = next_annotation_id
        incremented_anno_dict["id"] = next_annotation_id
        incremented_anno_dict["image_id"] = old_to_new_image_id_mapping[incremented_anno_dict["image_id"]]
        
        merged_instances_dict["annotations"].append(incremented_anno_dict)
        next_annotation_id += 1

### Create merged refs.json
# initialize merged instances using the first file
merged_refs_dict = {}
merged_instances_dict["info"] = {
    "description": f"This is dev 0.1 version of the 2024 RefCOCO 3D Synthetic Spatial dataset. \
                    Merged datasets {os.listdir(path_to_datasets)}",
    "version": "0.1",
    "year": 2024,
    "contributor": "University of Michigan",
    "date_created": f"{current_time_stamp}"
}
merged_instances_dict["images"] = instance_dictionaries_list[0]["images"]
merged_instances_dict["annotations"] = instance_dictionaries_list[0]["annotations"]
merged_instances_dict["categories"] = instance_dictionaries_list[0]["categories"]

last_image_id = merged_instances_dict["images"][-1]

# increment refs.json
for refs_dict in ref_dictionaries_list[1:]


# view changes
for image_dict in merged_instances_dict["images"]:
    
    print(image_dict)

for anno_dict in merged_instances_dict["annotations"]:
    
    show_anno = copy.deepcopy(anno_dict)
    show_anno["segmentation"] = []
    show_anno["pose"] = []
    print(show_anno)
                
        
        
            
    
        



# increement /images

    

