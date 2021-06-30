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

    names_to_int = {"aachen": 1, "bochum": 2, "bremen": 3, "cologne": 4, "darmstadt": 5, "dusseldorf": 6, "erfurt": 7,
                    "hamburg": 8, "hanover": 9, "jena": 10, "krefeld": 11, "monchengladbach": 12, "strasbourg": 13,
                    "stuttgart": 14, "tubingen": 15, "ulm": 16, "weimar": 17, "zurich": 18}

    anno_train = scipy.io.loadmat(anno_path + 'anno_train.mat')
    anno_train = anno_train['anno_train_aligned']

    for i in range(anno_train[0].shape[0]):
        # extract data from the annotations matrix
        city_name = anno_train[0, i][0][0][0][0]
        if not city_name == 'hamburg': # currently only uses hamburg 
            continue

        img_name = anno_train[0, i][0][0][1][0]
        p3 = img_name[-22:-16]
        p2 = img_name[-29:-23]
        p1=names_to_int[im[:-30]]
        p = str(p1) + p2 + p3

        one_im = {"id": int(p), #find out if we have image id
           "width": 2000, #do we have the same height/width and if not where is this information
           "height": 1000,
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
                           "image_id": int(p),  #same as above
                           "category_id": 1,
                           "segmentation":[[int(bb[1]), int(bb[2])]],
                           "area": int(bb[3] * bb[4]),
                           "height": int(bb[4]),
                           "vis_ratio": int(100*(bb[8]*bb[9])/(bb[3]*bb[4])),
                           "bbox": [int(bb[1]), int(bb[2]), int(bb[3]), int(bb[4])],
                           "iscrowd":0} #should probably be 0 for all instances
                ann.append(one_ann)


    data = {"info": inf, "licenses": lic, "images": im, "annotations": ann, "categories": cat}
    f = open("data.json","w")
    f. write(json.dumps(data))
