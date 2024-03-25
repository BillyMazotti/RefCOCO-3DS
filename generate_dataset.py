import sys,os,time

paths = ['/Users/billymazotti/miniforge3/lib/python3.9/site-packages/',
         os.getcwd()]

print()
print(os.path.dirname(sys.executable))

for path in paths:
    sys.path.append(path)

import bpy
import numpy as np
import cv2
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import json


import random
import math
from math import pi
from mathutils import Euler, Color, Vector
from pathlib import Path


# Make sure changes to local functions are being accounted for
import rotated_rect
from importlib import reload
reload(rotated_rect)
from rotated_rect import RRect_center




        
def positionCamera(x_pos, y_pos, z_pos, roll_deg):
    """
    Moves the CameraTarget Object 
    """
#    bpy.data.objects["Camera"].select_set(True)
#    current_state = bpy.data.objects["Camera"].select_get()
    bpy.context.view_layer.objects.active = bpy.data.objects['Camera']

    # position camera to specified x,y,z location
    bpy.data.objects["Camera"].location = [x_pos, y_pos, z_pos]
    
    # set specific axis to point up
    bpy.ops.object.constraint_add(type='TRACK_TO')
    bpy.data.objects["Camera"].constraints["Track To"].target = bpy.data.objects["CameraTarget"]
    bpy.data.objects["Camera"].constraints["Track To"].up_axis = 'UP_Y'
    bpy.ops.constraint.apply(constraint="Track To", owner='OBJECT')

    # save the upright angle
    upright_camera_orientation = bpy.data.objects["Camera"].rotation_euler
    bpy.context.scene.transform_orientation_slots[0].type = 'GLOBAL'
    upright_camera_orientation = bpy.data.objects["Camera"].rotation_euler
    
    # delete TRACK_TO constraint to allow for camera roll
    bpy.ops.constraint.delete(constraint="Track To", owner='OBJECT')
    bpy.data.objects["Camera"].rotation_euler = upright_camera_orientation
    
    # add roll to camera
    bpy.data.objects["Camera"].rotation_mode = 'ZYX'
    bpy.data.objects["Camera"].rotation_euler[2] = upright_camera_orientation[2] + \
                                                        roll_deg*pi/180



def placeCameraInVolume(cubeMeshName,roll):
    """
    randomly place camera in an UNROTATED mesh cube primitive
    
    cube primite cannot be rotated, only changed in scale and location
    """
    
    # define the x,y,z limits of the cube mesh
    centroidXYZ = np.array(bpy.data.objects[cubeMeshName].location)
    dimensionsXYZ = np.array(bpy.data.objects[cubeMeshName].dimensions)
    camera_volume_limits = centroidXYZ * np.ones((2,3))
    camera_volume_limits[0,:] -= dimensionsXYZ/2
    camera_volume_limits[1,:] += dimensionsXYZ/2

    camera_volume_limits_mm = (camera_volume_limits * 1000).astype(int)

    # genrate random x,y,z point within the volume for camera placement
    randX = random.randrange(start = camera_volume_limits_mm[0,0],
                            stop = camera_volume_limits_mm[1,0],
                            step = 1) / 1000
    randY = random.randrange(start = camera_volume_limits_mm[0,1],
                            stop = camera_volume_limits_mm[1,1],
                            step = 1) / 1000
    randZ = random.randrange(start = camera_volume_limits_mm[0,2],
                            stop = camera_volume_limits_mm[1,2],
                            step = 1) / 1000
    
    positionCamera(randX, randY, randZ, roll)

def generate_random_object_pose(object_plane_limits_mm,centroidXYZ,objects_dict,objectName):
    
    # genrate random x,y,z point within the volume for camera placement
    randX = random.randrange(start = object_plane_limits_mm[0,0],
                            stop = object_plane_limits_mm[1,0],
                            step = 1) / 1000
    randY = random.randrange(start = object_plane_limits_mm[0,1],
                            stop = object_plane_limits_mm[1,1],
                            step = 1) / 1000    
    z_pos = centroidXYZ[2]                    
            
    # generate random orientation wihting rotaiton limits
    if (objects_dict[objectName]["rot_limits"][0][1] - \
            objects_dict[objectName]["rot_limits"][0][0] == 0):
        randX_theta = objects_dict[objectName]["rot_limits"][0][0]
    else:
        randX_theta = random.randrange(start = objects_dict[objectName]["rot_limits"][0][0],
                                        stop = objects_dict[objectName]["rot_limits"][0][1],
                                        step = 1)
    if (objects_dict[objectName]["rot_limits"][1][1] - \
            objects_dict[objectName]["rot_limits"][1][0] == 0):
        randY_theta = objects_dict[objectName]["rot_limits"][0][0]
    else:
        randY_theta = random.randrange(start = objects_dict[objectName]["rot_limits"][1][0],
                                        stop = objects_dict[objectName]["rot_limits"][1][1],
                                        step = 1)
    if (objects_dict[objectName]["rot_limits"][2][1] - \
            objects_dict[objectName]["rot_limits"][2][0] == 0):
        randZ_theta = objects_dict[objectName]["rot_limits"][0][0]
    else:
        randZ_theta = random.randrange(start = objects_dict[objectName]["rot_limits"][2][0],
                                        stop = objects_dict[objectName]["rot_limits"][2][1],
                                        step = 1)
        
    return randX, randY, z_pos, randX_theta, randY_theta, randZ_theta

def add_polygons(contour_of_interest,polygon_list):
    
    # add polygon if first
    if len(polygon_list) == 0:
        polygon_list += (contour_of_interest,)
        # print("First Polygon Accepted")
        return polygon_list, True
    
    # check to see if contour_of_interst intersects with any contours in polygon_list
    candidate_polygon = Polygon(contour_of_interest.reshape(-1,2))
    
    for i in range(len(polygon_list)):
        polygon_i = polygon_list[i].reshape(-1,2)
        
        ### DEBUGGING OCCLUSIONS
        # polygon_i_visual = np.vstack((polygon_i,polygon_i[0,:]))
        # coi_visual = contour_of_interest.reshape(-1,2)
        # coi_visual = np.vstack((coi_visual,coi_visual[0,:]))
        
        # plt.figure()
        # plt.plot(polygon_i[:,0],polygon_i[:,1])
        # plt.plot(coi[:,0],coi[:,1])
        # ax = plt.gca()
        # ax.set_aspect('equal', adjustable='box')
        # plt.savefig('test_image.png')
        
        if candidate_polygon.intersects(Polygon(polygon_i)):
            return polygon_list, False
    
    # if we looked through all polgons without intersection, add polygon to list
    polygon_list += (contour_of_interest,)
    return polygon_list, True

def placeObjectOnPlane(planeName, objectName, objects_dict, placed_object_footprints, max_attempts=10):
    """
        assmes object is small enough to fit in plane
        assumes object has local z vecotr pointing up 
        
        current errors:
            - roll grows exponentially
            
    """
    
    # set position to ensure object is not hanging off plane
    # define the x,y limits of the plane
    centroidXYZ = np.array(bpy.data.objects[planeName].location)
    planeDimensionsXYZ = np.array(bpy.data.objects[planeName].dimensions)
    
    # look through all subparts to find 
    objectDimensionsXYZ = np.array(bpy.data.objects[objectName].dimensions)
   
    max_object_length = max(objectDimensionsXYZ[0],objectDimensionsXYZ[1])
    
    object_plane_limits = centroidXYZ[0:2] * np.ones((2,2))
    object_plane_limits[0,:] -= (planeDimensionsXYZ[0:2] - max_object_length)/2
    object_plane_limits[1,:] += (planeDimensionsXYZ[0:2] - max_object_length)/2
    
    object_plane_limits_mm = (object_plane_limits * 1000).astype(int)
    
    attempt_iter = 0
    while attempt_iter < max_attempts:
        # propose random object pose [m]
        randX, randY, z_pos, randX_theta, randY_theta, randZ_theta = \
            generate_random_object_pose(object_plane_limits_mm,centroidXYZ,objects_dict,objectName)
        
        # Compute Footprints
        (W_mm,H_mm) = (objectDimensionsXYZ[0]*1000,objectDimensionsXYZ[1]*1000)
        ang = -randZ_theta #degrees
        P0_mm = (randX*1000,randY*1000)
        rr = RRect_center(P0_mm,(W_mm,H_mm),ang)
        contour_mm = np.array([[[rr.verts[0][0],rr.verts[0][1]]],
                                [[rr.verts[1][0],rr.verts[1][1]]],
                                [[rr.verts[2][0],rr.verts[2][1]]],
                                [[rr.verts[3][0],rr.verts[3][1]]]])
        
        placed_object_footprints, new_polygon_added = add_polygons(contour_mm,placed_object_footprints)
        
        if new_polygon_added:
            # check to see if pose is valid with respect to currently placed objects
            bpy.data.objects[objectName].location = [randX, randY, z_pos]                             
            bpy.data.objects[objectName].rotation_euler = [randX_theta*math.pi/180,
                                                            randY_theta*math.pi/180,
                                                            randZ_theta*math.pi/180]
            # print(f"Iter {attempt_iter}: Accepted Polygon")
            break
        
        # print(f"Iter {attempt_iter}: Rejected Polygon")
        attempt_iter += 1
    
    return placed_object_footprints

def objects_in_fov():
    """
    Retern a list of all coco object namesg
    output:

    """
    
    camera = bpy.context.scene.camera
    fov = camera.data.angle
    location = camera.location
    direction = camera.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
    visible_objects = [obj for obj in bpy.context.scene.objects if not obj.hide_render]

    # creeat list of all visible objects in fob
    objects_in_fov = []
    for obj in visible_objects:
        if obj.type != 'MESH' or not obj.visible_get(): 
            continue    # skip non mesh and non-visible objects
        if obj.name.split("_")[0] != 'obj':
            continue    # skip objects that don't start with obj_
        for v in obj.data.vertices:
            vertex_world = obj.matrix_world @ v.co
            to_vertex = vertex_world - location
            angle_to_vertex = direction.angle(to_vertex)
            if angle_to_vertex < fov / 2:
                objects_in_fov.append(obj.name)
                break
            
    objects_in_fov.sort()
    return objects_in_fov



def annotate_2Dand_3D_data_of_in_view_objects():
    """
    Determine which objects are in view and are showing at least X pixels? in FOV?
    - Step 1: find all objects in fov using fov script (still could be occluded)
    - Step 2: create mask of each object in FOV and if there's at least 1 pixel then annotate the object
        https://www.youtube.com/watch?v=xeprI8hJAH8
    - Step 3: get the object's distance from the camera using object's centroid location 
        (make sure this is the actual centroid and not just the object's groudnign point) xyz data
    """
    
    
    # retreive objects in the field of view
    object_in_fov_names = objects_in_fov()
    
    # assign pass index to each object
    for IndexOB,obj in enumerate(object_in_fov_names):
        IndexOB += 1
        bpy.data.objects[obj].pass_index = IndexOB
        
    # get each objects 3D location wrt the camera
    
def move_away_all_objects(location):
    for obj in bpy.data.objects:
        if obj.name.split("_")[0] == "obj": 
            bpy.data.objects[obj.name].location = location
            bpy.data.objects[obj.name].rotation_euler[2] = 0.0

def relative_location(obj1_name, obj2_name):
    
    """
    inputs:
        obj1: primary object
        obj2: the object whose location we want with respect to obj1's local frame
    outputs:
        relative_location = [x, y, z]
    """
    cam = bpy.data.objects[obj1_name]
    obj = bpy.data.objects[obj2_name]
    mat_rel = cam.matrix_world.inverted() @ obj.matrix_world
    relative_location = -np.array(mat_rel.translation)

    return relative_location  

def deselect_all_objects():
    """ Deselect all objects
    """
    # deselect all objects
    bpy.ops.object.select_all(action='DESELECT')  

def select_object(obj_name):
    """ Select a specific object; ensure that no other objecrts are
        selected

    Args:
        obj_name (string): name a object to be selected; found
                        using bpy.data.objects["object_name"].name
    """
    deselect_all_objects()
    # select object of interest
    bpy.data.objects[obj_name].select_set(True) 
    current_state = bpy.data.objects[obj_name].select_get()
    bpy.context.view_layer.objects.active = bpy.data.objects[obj_name]

def duplicate_object(obj_name):
    """ Duplicate an object; name should end with .###
        (e.g. if original object ot be duplicated is named "pepsi_can"
        then the duplicated object will have the name "pepsi_can.001")

    Args:
        obj_name (string): name a object to be duplicated

    Returns:
        string: name of the newly duplicated object
    """
    select_object(obj_name)
    bpy.ops.object.duplicate()
    new_object_name = bpy.context.selected_objects[0].name
    
    return new_object_name
    
def delete_all_duplicate_objects():
    """ Delete all duplicated objects in the entire Scene Collection
    
        Assumptions:
        - all objects to be deleted must have names starting with obj_
        - all original objects (non-duplicates) must not have a "." in 
        their names
        - all duplicates must have a since "." in their names
    """
    for obj in bpy.data.objects:
        if obj.name.split("_")[0] == 'obj' and len(obj.name.split("."))==2:
            select_object(obj.name)
            bpy.ops.object.delete()
                    
                    
def dictionary_for_object_plane(objects_json_file_path):
    """ Generate Dictinary for Specified Object Plane

    Args:
        objects_json_file_path (string): path to json file
        with object plane specifications 

    Returns:
        dict: dictionary of all objects and their rotation
        limits
    """
    
    with open(objects_json_file_path) as json_file:
        objects_dict_original = json.load(json_file)
    
    objects_dict_for_object_plane = {}
    
    # generate duplicate objects specified by json file
    for obj in objects_dict_original:
        if objects_dict_original[obj]["max_number"] > 0:
            
            # add the original to the object plane's dictionary
            objects_dict_for_object_plane[obj] = {}
            objects_dict_for_object_plane[obj]["rot_limits"] = objects_dict_original[obj]["rot_limits"]
            
            # add duplicates to the object plane's dictionary
            number_of_objects_to_duplicate = objects_dict_original[obj]["max_number"] - 1
            for i in range(number_of_objects_to_duplicate):
                new_object_name = duplicate_object(obj)
                
                # add to dictionary (will eventually replace the list)
                objects_dict_for_object_plane[new_object_name] = {}
                objects_dict_for_object_plane[new_object_name]["rot_limits"] = objects_dict_original[obj]["rot_limits"]
                
        else:
            pass
        
    return objects_dict_for_object_plane


### TODO: Define Camera Volumes ###########################################


camera_volumes = ["CameraVolume1",
                  "CameraVolume2"]


###########################################################################

# select camera volume probabilistically based on volume
volume_sizes = np.zeros((len(camera_volumes)))
for i,volume in enumerate(camera_volumes):
    volume_sizes[i] = np.array(bpy.data.objects[volume].dimensions).prod()

volume_size_percentages = volume_sizes / volume_sizes.sum()
camera_volume_selected = np.random.choice(np.arange(0,len(camera_volumes),1), 
                                              size=1, replace=True, p=volume_size_percentages)[0]
placeCameraInVolume(camera_volumes[camera_volume_selected], roll=0)

# delete all duplicate objects and place all objects outside of envionrment
delete_all_duplicate_objects()
move_away_all_objects([5,5,0])

object_plane_dictionaries = {}
### TODO: define Object Plane dictionaries here ###########################


object_plane_dictionaries["ObjectPlane1"] = dictionary_for_object_plane("living_room_op1.json")
object_plane_dictionaries["ObjectPlane2"] = dictionary_for_object_plane("living_room_op2.json")


###########################################################################

placed_object_footprints = []
max_attemps_per_placement = 10
for object_plane in object_plane_dictionaries:
    object_plane_dict = object_plane_dictionaries[object_plane]
    for obj_name in object_plane_dict:
        placeObjectOnPlane(object_plane, 
                           obj_name, object_plane_dict, 
                           placed_object_footprints, 
                           max_attemps_per_placement)


pose = relative_location("Camera","obj_phone_0_ort_0")
print(pose)

# RENDER SETTINGS~
RENDER = False
bpy.data.scenes["Scene"].cycles.samples = 10

if RENDER:
    num_enviornmetns = 1
    start_idx = 0
    renders_per_environment = 1
    start_time = time.time()

    total_render_count = num_enviornmetns * renders_per_environment

    renter_rates = np.zeros(total_render_count)

    print("STARTING DATASET GENERATION...")
    for i in range(start_idx, start_idx + renders_per_environment):
        
        # randomly place camera in a volume defined by a cube mesh
        # placeCameraInVolume("CameraVolume",roll=0)
        
        # randomly place objects on plane defined by a plane mesh
        # TODO
        
        
        # set pass index for all objects to 0
        for obj in bpy.data.objects:
            bpy.data.objects[obj.name].pass_index = 0

        annotate_2Dand_3D_data_of_in_view_objects()

        # render image
        bpy.context.scene.render.filepath =  os.getcwd() + f"/data/images/{str(i).zfill(6)}.png"
        bpy.ops.render.render(write_still=True)
        
        
        img_rgb = cv2.imread(bpy.context.scene.render.filepath) 
        seg = cv2.imread("data/Segmentation0116.png")
        number_of_annotations = seg.max()

        for IndexOB in range(number_of_annotations):
            IndexOB+=1
            mask = np.zeros_like(seg)
            mask[seg == IndexOB] = 255

            imgray2 = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(imgray2, 0,255, cv2.THRESH_BINARY)
            contour,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # area = print(cv2.contourArea(contour[0]))
            cv2.drawContours(img_rgb, contour[0], -1, (0,255,0), 3)
        
        cv2.imshow(f"Image {str(i).zfill(6)}",img_rgb)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # save sementation image
        # os.rename(os.getcwd() + "/data/Segmentation0116.png", os.getcwd() + f"/data/masks/{str(i).zfill(6)}.png")
        
        # render rate statistics
        renter_rates[i] =  (time.time() - start_time) / (i + 1)
        seconds_remaining = renter_rates[i] * (total_render_count - i - 1)
        print(f'\nRemaining Time: {time.strftime("%H:%M:%S",time.gmtime(seconds_remaining))}s')
        print(f'Current | Avg | Max | Min Renter Rates (s/img): {round(renter_rates[i],2)} | {round(renter_rates[:i+1].mean(),2)} | {round(renter_rates[:i+1].max(),2)} | {round(renter_rates[:i+1].min(),2)}')

    print("DATASET GENERATION COMPLETE!")

# print("\nhello there\n")

    
    
        
