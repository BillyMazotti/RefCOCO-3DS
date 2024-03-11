# RefCOCO-Synthetic-3D



## Setting up Blender + VS Code programmiong Environment
RefCOCO Synthetic 3D Dataset Generation

For VS Code + Blender integration follow CG Python's tutorial ([Windows](https://www.youtube.com/watch?v=YUytEtaVrrc), [macOS](https://www.youtube.com/watch?v=_0srGXAzBZE), [Linux](https://www.youtube.com/watch?v=zP0s1i9EXeM)), although would recommend using Anaconda for python virtual environment.

Post tutorial, to quickly run new python code from blender with hotkey:
1) Open python script file in VS Code
2) Open VS Code's Command Palette (CTL/CMD + SHIFT + P), type in and select "Blender: Start"
4) In Blender, open up a new project or an in-progress project if you have one
5) To define a custom hotkey for running your script in VS Code, in VS Code's Command Palette, start typig in "Blend: Run Script" and click on the settings gear icon to the right of "Blend: Run Script" (see image below)
6) Change the "Keybinding" to whatever hotkey you want to use for running the script

<img width="491" alt="image" src="https://github.com/BillyMazotti/RefCOCO-Synthetic-3D/assets/96280520/f1b812cc-f343-44e3-a23b-842ad9d4db7d">


## RefCOCO-S3D dataset definition 
### RefCOCO's annotations format
```
instances.json = {
    'info': {
        'description': 'This is stable 1.0 version of the 2014 MS COCO dataset.', 
        'url': 'http://mscoco.org', 
        'version': '1.0', 
        'year': 2014, 
        'contributor': 'Microsoft COCO group', 
        'date_created': '2015-01-27 09:11:52.357475'
    },
    'images': [
        {'license': 1, 
        'file_name': 
        'COCO_train2014_000000098304.jpg', 
        'coco_url': 'http://mscoco.org/images/98304', 
        'height': 424, 
        'width': 640, 
        'date_captured': '2013-11-21 23:06:41', 
        'flickr_url': 'http://farm6.staticflickr.com/5062/5896644212_a326e96ea9_z.jpg', 
        'id': 98304},
        .
        .
        .
    ],
    'licenses': [
        {'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/', 
        'id': 1, 
        'name': 'Attribution-NonCommercial-ShareAlike License'}, 
        .
        .
        .
    ],
    'annotations': [
        {'segmentation': [[267.52, ..., 228.6]], 
        'area': 197.29899999999986, 
        'iscrowd': 0, 
        'image_id': 98304, 
        'bbox': [263.87, 216.88, 21.13, 15.17], 
        'category_id': 18, 'id': 3007},
        .
        .
        .
    ], 
    'categories': [
        {'supercategory': 'person', 'id': 1, 'name': 'person'}, 
        {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'}, 
        {'supercategory': 'vehicle', 'id': 3, 'name': 'car'}, 
        .
        .
        .
    ]
}
```

### Our annotations format
```
instances.json = {
    'info': {
        'description': 'This is stable 1.0 version of the 2024 MS COCO dataset.', 
        'url': 'http://mscoco.org', 
        'version': '1.0', 
        'year': 2014, 
        'contributor': 'Microsoft COCO group', 
        'date_created': '2015-01-27 09:11:52.357475'
    },
    'images': [
        {'license': 1, 
        'file_name': 
        'COCO_train2014_000000098304.jpg', 
        'coco_url': 'http://mscoco.org/images/98304', 
        'height': 424, 
        'width': 640, 
        'date_captured': '2013-11-21 23:06:41', 
        'flickr_url': 'http://farm6.staticflickr.com/5062/5896644212_a326e96ea9_z.jpg', 
        'id': 98304},
        .
        .
        .
    ],
    'licenses': [
        {'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/', 
        'id': 1, 
        'name': 'Attribution-NonCommercial-ShareAlike License'}, 
        .
        .
        .
    ],
    'annotations': [
        {'segmentation': [[267.52, ..., 228.6]], 
        'area': 197.29899999999986, 
        'iscrowd': 0, 
        'image_id': 98304, 
        'bbox': [263.87, 216.88, 21.13, 15.17], 
        'category_id': 18, 'id': 3007},
        .
        .
        .
    ], 
    'categories': [
        {'supercategory': 'person', 'id': 1, 'name': 'person'}, 
        {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'}, 
        {'supercategory': 'vehicle', 'id': 3, 'name': 'car'}, 
        .
        .
        .
    ]
}
```
