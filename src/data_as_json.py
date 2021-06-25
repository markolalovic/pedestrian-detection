import json
import scipy.io
from datetime import date

def data_as_json(anno_path):

    inf = {"year": 2021,
           "version": 1,
            "describtion": "Data for object detection",
            "contributor": "ML-group",
            "url": "test", #leave all urls empty?
            "date_created": str(date.today())}

    #can we leave licences like this?
    lic = [{"id": 1,
            "name": "test",
            "url": "test"}]

    cat = [{"id": 1,
            "name": "pedestrian",
            "supercategory": "person"}]

    im = []
    ann = []

    anno_train = scipy.io.loadmat(anno_path + 'anno_train.mat')
    anno_train = anno_train['anno_train_aligned']

    for i in range(anno_train[0].shape[0]):
        # extract data from the annotations matrix
        city_name = anno_train[0, i][0][0][0][0]
        if not city_name == 'hamburg': # currently only uses hamburg 
            continue

        img_name = anno_train[0, i][0][0][1][0]

        one_im = {"id": i+1, #find out if we have image id
           "width": 640, #do we have the same height/width and if not where is this information
           "height": 640,
           "file_name": str(img_name),
           "license": 1,
           "flickr_url": "test",
           "coco_url": "test",
           "date_captured": "test"} #probably also irrelevant

        im.append(one_im)

        for bb in anno_train[0, i][0][0][2]:
            ## format is: [class_label, x1,y1,w,h, instance_id, x1_vis, y1_vis, w_vis, h_vis]
            if bb[0] == 1:  # class_label = 1 means it is a pedestrian
                one_ann = {"id": int(bb[5]),
                           "image_id": i+1,  #same as above
                           "category_id": 1,
                           "segmentation":[list(bb[1:3])],
                           "area": int(bb[3] * bb[4]),
                           "bbox": list(bb[1:5]),
                           "iscrowd":0} #should probably be 0 for all instances
                ann.append(one_ann)


    data = {"info": inf, "licenses": lic, "images": im, "annotations": ann, "categories": cat}
    f = open("data.json","w")
    f. write(json.dumps(data))
