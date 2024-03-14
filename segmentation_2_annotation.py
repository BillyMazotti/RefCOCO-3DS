import numpy as np
import cv2
import os
import json
from json_formatting import NoIndent, MyEncoder
    

mask_data = []

number_of_masks = len(os.listdir(os.getcwd() + "/data/masks"))


for image_idx in range(number_of_masks):
    
    img_seg = cv2.imread(f"data/masks/{str(image_idx).zfill(6)}.png")
    
    mask_data_dict_i = {}
    mask_data_dict_i["filename"] = f"{str(image_idx).zfill(6)}.png"
    mask_data_dict_i["height"] = img_seg.shape[0]
    mask_data_dict_i["width"] = img_seg.shape[1]
    mask_data_dict_i["detection"] = {}
    mask_data_dict_i["detection"]["instances"] = []
    
    mask_data_dict_i["grounding"] = {}
    mask_data_dict_i["grounding"]["caption"] = None
    mask_data_dict_i["grounding"]["regions"] = []
    
    
    
    number_of_annotations = img_seg.max()
    
    for IndexOB in range(number_of_annotations):
        IndexOB+=1
        
        img_rgb = cv2.imread(f"data/images/{str(image_idx).zfill(6)}.png")

        mask = np.zeros_like(img_seg)
        mask[img_seg == IndexOB] = 255

        imgray2 = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray2, 0,255, cv2.THRESH_BINARY)
        contour,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # area = print(cv2.contourArea(contour[0]))
        # cv2.drawContours(img_rgb, contour, -1, (0,255,0), 3)
        # cv2.imshow(f"Image {str(image_idx).zfill(6)}",img_rgb)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # add segmentaiton and bbox data
        detection_instance_data = {}
        groudning_region_data = {}
        
        # if we have multiple contours
        if len(contour) != 1:
            print(f"{len(contour)} contours found for object")
            detection_instance_data["segmentation"] = []
            detection_instance_data["bbox"] = []
            groudning_region_data["segmentation"] = []
            groudning_region_data["bbox"] = []
        else:
        # add instance annotation information
            contour_list = contour[0].reshape(-1,2).tolist()
            detection_instance_data["segmentation"] = NoIndent(contour_list)
            groudning_region_data["segmentation"] =  NoIndent(contour_list)
            (x,y,w,h) = cv2.boundingRect(contour[0])
            detection_instance_data["bbox"] = NoIndent([x,y,w,h])
            groudning_region_data["bbox"] = NoIndent([x,y,w,h])
            
            
        # additional empty values
        detection_instance_data["label"] = None
        detection_instance_data["category"] = None
        groudning_region_data["phrase"] = None
        
        mask_data_dict_i["detection"]["instances"].append(detection_instance_data)
        mask_data_dict_i["grounding"]["regions"].append(groudning_region_data)
        
    
    # add image data to output
    mask_data.append(mask_data_dict_i)


json_object = json.dumps(mask_data, cls=MyEncoder, sort_keys=False, indent=2)
# # Writing to sample.json
with open("sample.json", "w") as outfile:
    outfile.write(json_object)
    
    
    
    
# Remaining information that we need
"""
detection.label --> either the nth annotation in the image or the numerical vlaue forthe class category
detection.category --> class name, single word typically
grounding.caption --> caption for the entire image
grounding.regions.phrase --> An single phrase for the the region


Questions:
- what exactly is detection.label?
- do we need to have an entire caption for the image? --> let's assume no for now and cross that bridge when we get there
- are we able to hagve more than one phrase per region like the refcoco examples?
"""