import os
import json
from datetime import datetime
import copy
from tqdm import tqdm
import shutil

""" Merge datasets in specified directory
    assumes all datasets have the same "cataegories" key and aassociated field values
"""

path_to_datasets = os.getcwd() + "/saved_datasets/"

# create lists of all dictionaries instances.json and refs.json
instance_dictionaries_list = []
ref_dictionaries_list = []
ref_coco_dictionaries = [dataset for dataset in os.listdir(path_to_datasets) if dataset.split("_")[0] == "RefCOCO"]
for dataset in ref_coco_dictionaries:
    with open(path_to_datasets + dataset + "/instances.json") as json_file:
        instance_dictionaries_list.append(json.load(json_file))
    with open(path_to_datasets + dataset + "/refs.json") as json_file:
        ref_dictionaries_list.append(json.load(json_file))

# create the merged dataset directory
current_time_stamp = "merged_datasets_" + str(datetime.now()).replace(":","_")
merged_dataset_directory = os.getcwd() + "/merged_datasets/" + current_time_stamp

if(os.path.exists(merged_dataset_directory)):
    print(f"Merged Dataset {os.getcwd() + '/merged_datasets/' + current_time_stamp} already exists")
else:
    os.mkdir(merged_dataset_directory)
    os.mkdir(merged_dataset_directory + "/images")


# initialize dump values for instances.json and refs.json
# instances.json
merged_instances_dict = {}
merged_instances_dict["info"] = {
    "description": f"This is dev 0.1 version of the 2024 RefCOCO 3D Synthetic Spatial dataset. \
                    Merged datasets {os.listdir(path_to_datasets)}",
    "version": "0.1",
    "year": 2024,
    "contributor": "University of Michigan",
    "date_created": f"{current_time_stamp}"
}   # will not require further editing
merged_instances_dict["images"] = instance_dictionaries_list[0]["images"]
merged_instances_dict["annotations"] = instance_dictionaries_list[0]["annotations"]
merged_instances_dict["categories"] = instance_dictionaries_list[0]["categories"]   # will not require further editing
# refs.json
merged_refs_list = []
merged_refs_list += ref_dictionaries_list[0]


### Merged the instances.json files ##########################

last_image_id = merged_instances_dict["images"][-1]
# increment instnaces.json - images
next_image_id = merged_instances_dict["images"][-1]["id"] + 1
old_to_new_image_id_mapping = {}
next_annotation_id = merged_instances_dict["annotations"][-1]["id"] + 1
old_to_new_annotation_id_mapping = {}

print("\nMerging instances.json files...")
for dataset_idx, instances_dict in enumerate(tqdm(instance_dictionaries_list[1:])):
    dataset_idx += 1
    old_to_new_image_id_mapping[ref_coco_dictionaries[dataset_idx]] = {}
    old_to_new_annotation_id_mapping[ref_coco_dictionaries[dataset_idx]] = {}
    
    # increment all the images
    for image_dict in instances_dict["images"]:
        
        old_to_new_image_id_mapping[ref_coco_dictionaries[dataset_idx]][image_dict["id"]] = next_image_id
        incremented_image_dict = copy.deepcopy(image_dict)
        incremented_image_dict["file_name"] = str(next_image_id).zfill(6) + ".png"
        incremented_image_dict["id"] = next_image_id
        
        merged_instances_dict["images"].append(incremented_image_dict)
        
        next_image_id += 1
    
    # increement all the annotations
    for anno_dict in instances_dict["annotations"]:
        
        old_to_new_annotation_id_mapping[ref_coco_dictionaries[dataset_idx]][anno_dict["id"]] = next_annotation_id
        incremented_anno_dict = copy.deepcopy(anno_dict)
        incremented_anno_dict["id"] = next_annotation_id
        incremented_anno_dict["image_id"] =  old_to_new_image_id_mapping[ref_coco_dictionaries[dataset_idx]][incremented_anno_dict["image_id"]]
        
        merged_instances_dict["annotations"].append(incremented_anno_dict)
        next_annotation_id += 1
print("Merging instances.json files... Complete!")



### Merged the refs.json files ##############################

# need to increment: sent_ids, file_name, ann_id, ref_id, image_id, sent_id
next_sent_id = merged_refs_list[-1]["sent_ids"][-1]
print("Merging refs.json files...")
for dataset_idx, refs_list in enumerate(tqdm(ref_dictionaries_list[1:])):
    dataset_idx += 1
    for anno_dict in refs_list:
       
        incremented_anno_dict = copy.deepcopy(anno_dict)
       
        # increment the send_ids in "sent_ids" and "sentences"
        for sent_id_idx in range(len(anno_dict["sent_ids"])):

            incremented_anno_dict["sent_ids"][sent_id_idx] = next_sent_id
            incremented_anno_dict["sentences"][sent_id_idx]["sent_id"] = next_sent_id
            
            next_sent_id += 1
            
        incremented_anno_dict["file_name"] = str(old_to_new_image_id_mapping[ref_coco_dictionaries[dataset_idx]][anno_dict["image_id"]]).zfill(6) + '.png'
        incremented_anno_dict["image_id"] = old_to_new_image_id_mapping[ref_coco_dictionaries[dataset_idx]][anno_dict["image_id"]]
        incremented_anno_dict["ann_id"] = old_to_new_annotation_id_mapping[ref_coco_dictionaries[dataset_idx]][anno_dict["ann_id"]]
        incremented_anno_dict["ref_id"] = incremented_anno_dict["ann_id"]   # ref_id is the same as ann_id
        
        merged_refs_list.append(incremented_anno_dict)
print("Merging refs.json files... Complete!")


print("Merging image files...")
dataset = ref_coco_dictionaries[0]
image_list = os.listdir(path_to_datasets + dataset + "/images")
# increment and send over image data
for image_name in image_list:
    image_path_src = path_to_datasets + dataset + "/images/" + image_name
    image_path_dst = merged_dataset_directory + "/images/" + image_name
    shutil.copyfile(image_path_src,image_path_dst)
        
for dataset in tqdm(ref_coco_dictionaries[1:]):
    
    image_list = os.listdir(path_to_datasets + dataset + "/images")
    for image_name in image_list:
        
        
        image_id_old = image_name.split(".")[0].lstrip("0")
        if image_id_old == "": image_id_old = 0
        new_image_name = str(old_to_new_image_id_mapping[dataset][int(image_id_old)]).zfill(6) + '.png'
        image_path_src = path_to_datasets + dataset + "/images/" + image_name
        image_path_dst = merged_dataset_directory + "/images/" + new_image_name
        shutil.copyfile(image_path_src,image_path_dst)
print("Merging image files... Complete!")


# print("Removing images calls for images that don't exist")

# print("Removing images calls for images that don't exist... Complete!")


print("Dumping data to instances.json and refs.json...")
with open(merged_dataset_directory+"/instances.json", 'w') as f:
    json.dump(merged_instances_dict, f)

with open(merged_dataset_directory+"/refs.json", 'w') as f:
    json.dump(merged_refs_list, f)
print("Dumping data to instances.json and refs.json... Complete!")


