import json
import sys
import numpy as np
import copy
from trope_utils import extract_key_objects, generate_proposals, aggregate_and_add_to_caption

def convert_box(bbox,current_format):
	if current_format == 'xywh':
		box_out = [bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]]
	else:
		box_out = [bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1]]

	return box_out

def get_img_files(dataset):
	base_dir = 'detector_info/'+dataset+'/'
	dataset_base = 'datasets/'+dataset+'/'
	img_fh = open(base_dir+'images_test.txt','r')
	id_to_file = {}
	for line in img_fh.readlines():
		img_info = line.rstrip().split(' ')
		id_to_file[int(img_info[0])] = dataset_base+img_info[1]
	img_fh.close()
	return id_to_file

def get_detector_info(base_dir,dataset):
	det_cats = 'VG-SGG-dicts-vgoi6-clipped.json'
	fh = open(base_dir+det_cats, 'r')
	det_info = json.load(fh)
	fh.close()
	base_dir += dataset+'/'
	idx_to_lab = det_info['idx_to_label']
	idx_to_attr = det_info['idx_to_attribute']
	num_lab_files = 5
	img_fh = open(base_dir+'images_test.txt','r')
	test_ids = []
	for line in img_fh.readlines():
		img_info = line.rstrip().split(' ')
		img_id = img_info[0]
		test_ids.append(img_id)

	bboxes = {int(img_id):[] for img_id in test_ids}
	labs = {int(img_id):[] for img_id in test_ids}
	attr_labs = {int(img_id):[] for img_id in test_ids}
	attr_scores = {int(img_id):[] for img_id in test_ids}
	scores = {int(img_id):[] for img_id in test_ids}

	for i in range(0,num_lab_files):
		with open(base_dir+dataset+'_bboxes'+str(i)+'.json','r') as fh:
			bbox_set = json.load(fh)
			fh.close()
		with open(base_dir+dataset+'_labels'+str(i)+'.json', 'r') as fh:
			lab_set = json.load(fh)
			fh.close()
		with open(base_dir+dataset+'_attr_labels'+str(i)+'.json','r') as fh:
			attr_lab_set = json.load(fh)
			fh.close()
		with open(base_dir+dataset+'_attr_scores_'+str(i)+'.json','r') as fh:
			attr_score_set = json.load(fh)
			fh.close()
		with open(base_dir+dataset+'_scores'+str(i)+'.json','r') as fh:
			score_set = json.load(fh)
			fh.close()
		for img_id in bbox_set.keys():
			bboxes[int(img_id)].extend([convert_box([float(val) for val in box],'xywh') for box in bbox_set[img_id]])
			labs[int(img_id)].extend([idx_to_lab[z] for z in lab_set[img_id]])
			attr_labs[int(img_id)].extend([idx_to_attr[z[0]] for z in attr_lab_set[img_id]])
			attr_scores[int(img_id)].extend([z[0] for z in attr_score_set[img_id]])

			scores[int(img_id)].extend([z for z in score_set[img_id]])
	return bboxes, labs, attr_labs, attr_scores, scores


dataset = sys.argv[1]
num_props = [1,5]#[2,3,4,6,7,8,9,10]

frac_thres = float(sys.argv[2])
detector_dir = 'detector_info/'
caption_dir = 'base_captions/'
model = sys.argv[3]#'baseline'

num_lab_files = 5
base_cap = {}
with open(caption_dir+'captions_out_'+dataset+'_'+model+'.json', 'r') as fh:
	base_cap = json.load(fh)
	base_cap = {int(entry['image_id']):entry['caption'] for entry in base_cap}
	fh.close()

enhanced_cap = {num_prop:{img_id:'' for img_id in base_cap.keys()} for num_prop in num_props}

bboxes, labs, attr_labs, attr_scores, scores = get_detector_info(detector_dir,dataset)

for img_id in base_cap.keys():
	key_objects, key_object_locs, caption_objects, object_boxes, cap_toks = extract_key_objects(base_cap[img_id], 
																					  			bboxes[img_id], 
																					  			labs[img_id])

	part_props = generate_proposals(key_objects, 
									caption_objects, 
									object_boxes, 
									bboxes[img_id], 
									labs[img_id], 
									scores[img_id], 
									attr_labs[img_id], 
									frac_thres)

	loc_order = np.argsort(key_object_locs)
	objs_orig = [key_objects[i] for i in loc_order]
	obj_locs_orig = [key_object_locs[i] for i in loc_order] 
	for num_prop in num_props:
		ordered_key_objs = copy.copy(objs_orig)
		ordered_obj_locs = copy.copy(obj_locs_orig)
		enhanced_cap[num_prop][img_id] = aggregate_and_add_to_caption(enhanced_cap[num_prop][img_id], 
																	  ordered_key_objs, 
																	  ordered_obj_locs, 
																	  part_props,
																	  cap_toks, 
																	  num_prop)

for num_prop in num_props:
	cap_out = []
	for entry in enhanced_cap[num_prop].keys():
		cap_out.append({'image_id':str(entry),'caption':enhanced_cap[num_prop][entry]}) 

	with open('results/captions_out_'+dataset+'_'+model+'_TROPE_'+str(num_prop)+'parts_frac'+str(frac_thres)+'_new.json', 'w') as fh:
		json.dump(cap_out,fh)	
		fh.close()
