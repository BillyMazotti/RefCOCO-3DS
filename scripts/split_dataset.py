# Split coco dataset (excluding licensing information)
import os
import json
import numpy as np
from datetime import datetime
import shutil
from tqdm import tqdm

def load_instances_and_refs(dataset_path):

    with open(dataset_path + "/instances.json") as json_file:
        intsances_dict = json.load(json_file)
        
    with open(dataset_path + "/refs.json") as json_file:
        refs_list = json.load(json_file)

    
        
    return intsances_dict, refs_list

def creat_dataset_directory(split_dataset_directory):
    
    # create the necessary folders
    
    if(os.path.exists(split_dataset_directory)):
        print(f"Directory {split_dataset_directory} already exists")
    else:
        os.mkdir(split_dataset_directory)
        os.mkdir(split_dataset_directory + "/train")
        os.mkdir(split_dataset_directory + "/train/images")
        os.mkdir(split_dataset_directory + "/val")
        os.mkdir(split_dataset_directory + "/val/images")
        os.mkdir(split_dataset_directory + "/test")
        os.mkdir(split_dataset_directory + "/test/images")
        
def generate_split_assignment_array(instances_dict_original, percentage_train, percentage_val):
    
    # determime which images should be train/val
    num_images = len(instances_dict_original["images"])
    num_train = int(num_images*percentage_train)    # 2
    num_val = int(num_images*percentage_val)        # 1
    num_test = num_images - num_train - num_val     # 0
    split_assignment_array = np.concatenate((np.zeros(num_test),np.ones(num_val),2*np.ones(num_train)))
    np.random.shuffle(split_assignment_array)
    mapping_image_idx_to_split_type = {}
    mapping_image_idx_to_split_type = {}
    for idx, val in enumerate(split_assignment_array):
        mapping_image_idx_to_split_type[idx] = val
        
    
    return split_assignment_array, mapping_image_idx_to_split_type

def initialie_instances_dict(original_dataset_name, current_time_stamp, instances_dict_original):
    
    instances_dict = {}
    instances_dict["info"] = {
        "description": f"This is dev 0.1 version of the 2024 RefCOCO 3D Synthetic Spatial dataset. \
                        Merged datasets {original_dataset_name}",
        "version": "0.1",
        "year": 2024,
        "contributor": "University of Michigan",
        "date_created": f"{current_time_stamp}"
    }  
    instances_dict["images"] = []
    instances_dict["annotations"] = []
    instances_dict["categories"] = instances_dict_original["categories"]
    
    return instances_dict


def add_image_to_split(split_name, image_id, original_dataset_image_directory, split_dataset_directory):
    
    image_name = str(image_id).zfill(6)
    image_path_src = original_dataset_image_directory + "/" + image_name+ ".png"
    image_path_dst = split_dataset_directory + f"/{split_name}/images/" + image_name + ".png"
    shutil.copyfile(image_path_src,image_path_dst)



### USER INPUT ############################################################
# load dataset
dataset_to_split_name = "merged_datasets_2024-04-16 19_01_53.929290"    # test with 540 images

percentage_train = 0.7
percentage_val = 0.2

###########################################################################


# dataset is presumed to come from the merged_dataset directory
local_dataset_path = "/merged_datasets/" + dataset_to_split_name
original_dataset_path = os.getcwd() + local_dataset_path
original_dataset_image_directory = original_dataset_path + '/images'

# define destination dataset path
current_time_stamp = datetime.now()
path_of_split_dataset = os.getcwd() + "/split_datasets/"
split_dataset_name = "RefCOCO_3ds_" + str(current_time_stamp).replace(":","_")
split_dataset_directory = path_of_split_dataset + split_dataset_name
creat_dataset_directory(split_dataset_directory)

#load original dataset
instances_dict_original, refs_list_original = load_instances_and_refs(original_dataset_path)

# initialize dump values for instances.json and refs.json
instances_dict_train = initialie_instances_dict(split_dataset_name, current_time_stamp, instances_dict_original)
instances_dict_val = initialie_instances_dict(split_dataset_name, current_time_stamp, instances_dict_original)
instances_dict_test = initialie_instances_dict(split_dataset_name, current_time_stamp, instances_dict_original)
refs_list_train = []
refs_list_val = []
refs_list_test = []

split_assignment_array, split_assigment_mapping = generate_split_assignment_array(instances_dict_original, percentage_train, percentage_val)


print("Tranfering image files and image data from instance.json...")
# add images and iamge information from instances.json
for image_idx, image_dict in enumerate(tqdm(instances_dict_original["images"])):
    
    image_id = image_dict["id"]
    split_type = split_assigment_mapping[image_dict["id"]]
    
    if split_type == 2:
        add_image_to_split("train", image_id, original_dataset_image_directory, split_dataset_directory)
        instances_dict_train["images"].append(image_dict)

    elif split_type == 1:
        add_image_to_split("val", image_id, original_dataset_image_directory, split_dataset_directory)
        instances_dict_val["images"].append(image_dict)
        
    elif split_type == 0:
        add_image_to_split("test", image_id, original_dataset_image_directory, split_dataset_directory)
        instances_dict_test["images"].append(image_dict)
        
    else:
        print("Unknown Split Type")
print("Tranfering image files and image data from instance.json... Complete!")


print("Tranfering Annotation Data from instance.json...")
# add annotation information from instances.json
for anno_idx, anno_dict in enumerate(tqdm(instances_dict_original["annotations"])):
                  
    split_type = split_assigment_mapping[anno_dict["image_id"]]
    
    if split_type == 2:
        instances_dict_train["annotations"].append(anno_dict)

    elif split_type == 1:
        instances_dict_val["annotations"].append(anno_dict)
        
    elif split_type == 0:
        instances_dict_test["annotations"].append(anno_dict)
        
    else:
        print("Unknown Split Type")
        
print("Tranfering Annotation Data from instance.json... Complete!")


print("Tranfering reference data from refs.json...")
# add reference information from refs.json
for ref_idx, ref_dict in enumerate(tqdm(refs_list_original)):
    
    split_type = split_assigment_mapping[ref_dict["image_id"]]
    
    if split_type == 2:
        refs_list_train.append(ref_dict)

    elif split_type == 1:
        refs_list_val.append(ref_dict)
        
    elif split_type == 0:
        refs_list_test.append(ref_dict)
        
    else:
        print("Unknown Split Type")
print("Tranfering reference data from refs.json... Complete!")


print("Dumping data to instances.json and refs.json for train, val, and test...")
# create json files
#train
with open(split_dataset_directory + "/train/instances.json", 'w') as f:
    json.dump(instances_dict_train, f)
with open(split_dataset_directory + "/train/refs.json", 'w') as f:
    json.dump(refs_list_train, f)
#val
with open(split_dataset_directory + "/val/instances.json", 'w') as f:
    json.dump(instances_dict_val, f)
with open(split_dataset_directory + "/val/refs.json", 'w') as f:
    json.dump(refs_list_val, f) 
#test
with open(split_dataset_directory + "/test/instances.json", 'w') as f:
    json.dump(instances_dict_test, f)
with open(split_dataset_directory + "/test/refs.json", 'w') as f:
    json.dump(refs_list_test, f)
print("Dumping data to instances.json and refs.json for train, val, and test... Complete!")
