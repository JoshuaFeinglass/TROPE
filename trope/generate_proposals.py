import csv
import json
from collections import namedtuple
import spacy
import numpy as np
import sys
import inflect
import copy
pluralize = inflect.engine()
adj_list = ['long','short','yellow','orange','green','red','rectangular','pink','blue','blond','blank','black and white','black','white']
Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
nlp = spacy.load("en_core_web_md")
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

def distance(p1,p2):
	return np.sqrt(np.square(p2[0]-p1[0])+np.square(p2[1]-p1[1]))

def centroid(r1):
	a = Rectangle(float(r1[0]),float(r1[1]),float(r1[2]),float(r1[3]))
	return [(a.xmax+a.xmin)/2, (a.ymax+a.ymin)/2]

def box_contains(r1,p1):
	a = Rectangle(float(r1[0]),float(r1[1]),float(r1[2]),float(r1[3]))
	return a.xmin <= p1[0] and p1[0] <= a.xmax and a.ymin <= p1[1] and p1[1] <= a.ymax 

def combine_rects(r1,r2):
	if r1 is None:
		return r2
	elif r2 is None:
		return r1

	return [min(r1[0],r2[0]),min(r1[1],r2[1]),max(r1[2],r2[2]),max(r1[3],r2[3])]

def inter_area(r1, r2):  # returns None if rectangles don't intersect
	a = Rectangle(float(r1[0]),float(r1[1]),float(r1[2]),float(r1[3]))
	b = Rectangle(float(r2[0]),float(r2[1]),float(r2[2]),float(r2[3]))
	dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
	dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
	if (dx>=0) and (dy>=0):
		return dx*dy
	else:
		return 0

def area(r1):
	a = Rectangle(float(r1[0]),float(r1[1]),float(r1[2]),float(r1[3]))
	return (a.xmax-a.xmin)*(a.ymax-a.ymin)

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


dataset = sys.argv[1]#'CUB'
num_props = [1,5]#[2,3,4,6,7,8,9,10]#int(sys.argv[2])#2

frac_thres = float(sys.argv[2])#float(sys.argv[2])#float(sys.argv[2])#0.5
#addfrac = (sys.argv[4].lower() == 'true')#2
detector_dir = 'detector_info/'
caption_dir = 'base_captions/'
model = sys.argv[3]#'baseline'#'conzic'#'baseline'

num_lab_files = 5
base_cap = {}
with open(caption_dir+'captions_out_'+dataset+'_'+model+'.json', 'r') as fh:
	#artifact from loading VINVL tsv files
	#for line in fh.readlines(): 
	#	entries = line.split('\t')
	#	base_cap[int(entries[0])] = json.loads(entries[1])[0]['caption']
	base_cap = json.load(fh)
	base_cap = {int(entry['image_id']):entry['caption'] for entry in base_cap}
	fh.close()

enhanced_cap = {num_prop:{img_id:'' for img_id in base_cap.keys()} for num_prop in num_props}

bboxes, labs, attr_labs, attr_scores, scores = get_detector_info(detector_dir,dataset)

parts = {ex:{} for ex in base_cap.keys()}

for img_id in base_cap.keys():
	toks = nlp(base_cap[img_id])
	stem_labs = set(entry.lower() for entry in labs[img_id])
	objs = []
	all_objs = []
	obj_locs = []
	check_plural = []
	for i,tok in enumerate(toks):
		tok_lowered = tok.text.lower()
		all_objs.append(tok_lowered)
		check_plural.append(pluralize.singular_noun(tok_lowered))
		if check_plural[-1] is False:
			stem_tok = tok_lowered
		else:
			stem_tok = check_plural[-1]
		if stem_tok in stem_labs and tok.pos_ == 'NOUN' and toks[i-1].pos_ != 'NOUN' and toks[i+1].pos_ != 'NOUN' and toks[i+1].text.lower() != 'of':
			objs.append(stem_tok)
			obj_locs.append(i)
			all_objs.append(stem_tok)

	obj_boxes = {}
	obj_areas = []
	for j,obj in enumerate(objs):	
		parts[img_id][obj] = []
		for i,entry in enumerate(labs[img_id]):
			if entry == obj:
				if check_plural[obj_locs[j]] is False and obj not in obj_boxes.keys():
					obj_boxes[obj] = bboxes[img_id][i]
				elif check_plural[obj_locs[j]] is False and obj in obj_boxes.keys():
					continue
				elif check_plural[obj_locs[j]] is not False and obj not in obj_boxes.keys():
					obj_boxes[obj] = None
					obj_boxes[obj] = combine_rects(obj_boxes[obj],bboxes[img_id][i])
				else:
					obj_boxes[obj] = combine_rects(obj_boxes[obj],bboxes[img_id][i])

		obj_areas.append(area(obj_boxes[obj]))

	objs = [objs[i] for i in np.argsort(obj_areas)]
	obj_locs = [obj_locs[i] for i in np.argsort(obj_areas)]


	added = set()
	for obj in objs:
		#to_add = {}
		to_add = set()
		for ind,bbox in enumerate(bboxes[img_id]):
			#stem?
			if labs[img_id][ind] in all_objs or labs[img_id][ind] in added:
				continue
			
			area_bbox = area(bbox)
			frac = inter_area(obj_boxes[obj],bbox)/area_bbox

			if frac >= frac_thres and labs[img_id][ind] not in adj_list:
				# if nlp(labs[img_id][ind])[0].pos_ == 'ADJ':
				# 	print(labs[img_id][ind])

				attr_to_add = attr_labs[img_id][ind] 
				
				article = 'a'
				if attr_to_add[0] in ['a','e','i','o','u']:
					article = 'an'

				to_add.add(labs[img_id][ind])

				part_tup = (article+' '+attr_to_add+' '+labs[img_id][ind],frac+float(scores[img_id][ind]))#np.mean([float(attr_scores[img_id][ind]),float(scores[img_id][ind])]))

				parts[img_id][obj].append(part_tup)

		added = added.union(to_add)
	
	loc_order = np.argsort(obj_locs)
	objs_orig = [objs[i] for i in loc_order]
	obj_locs_orig = [obj_locs[i] for i in loc_order] 
	for num_prop in num_props:
		objs = copy.copy(objs_orig)#[:num_prop]
		obj_locs = copy.copy(obj_locs_orig)#[:num_prop]
		prop_added = False
		for i,tok in enumerate(toks):
			if prop_added == True and (tok.text == 'with' or tok.text == 'has' or tok.text == 'have'):
				prop_added = False
				enhanced_cap[num_prop][img_id] += ' in addition to'
				continue 

			prop_added = False

			if tok.text == '\'s' or toks[i-1].text == '\'s':
				continue
			
			if i != 0 and tok.text not in  ['.',',',':',';'] :
				enhanced_cap[num_prop][img_id] += ' '

			enhanced_cap[num_prop][img_id] += tok.text.lower()
			if i<(len(toks)-1) and toks[i+1].text == '\'s':
				enhanced_cap[num_prop][img_id] += toks[i+1].text	
				enhanced_cap[num_prop][img_id] += ' '+toks[i+2].text	

			
			if len(obj_locs)>0 and obj_locs[0] == i:
				del obj_locs[0]
				obj = objs.pop(0)
				parts[img_id][obj] = [prop for prop in parts[img_id][obj] if prop is not None]
				if len(parts[img_id][obj])>0:
					prop_added = True
					prop_order = np.flipud(np.argsort([prop[1] for prop in parts[img_id][obj]]))
					prop_ordered = [parts[img_id][obj][entry][0] for i,entry in enumerate(prop_order)]

					
					prop_aggregated = []
					terms_agg = {}
					terms_descr = {}
					for i,prop in enumerate(prop_ordered):
						prop_split = prop.split(' ')
						term = ' '.join(prop_split[2:])
						if term not in terms_agg.keys():
							terms_agg[term] = []
							terms_descr[term] = set()
						if prop_split[1] not in terms_descr[term] and len(terms_agg[term]) < num_prop:
							terms_agg[term].append(i)
							terms_descr[term].add(prop_split[1])
					for j,prop in enumerate(prop_ordered):
						prop_split = prop.split(' ')
						term = ' '.join(prop_split[2:])
						if terms_agg[term][0] != j:
							continue 
						if len(terms_agg[term])>1:
							attr_text = ''
							added = set()
							for i,entry in enumerate(terms_agg[term]):
								prop_attr = prop_ordered[entry].split(' ')[1]
								if i == 0:
									attr_text += prop_attr
								elif i == len(terms_agg[term])-1:
									if len(terms_agg[term]) == 2:
										attr_text += ' and '+prop_attr
									else:
										attr_text += ', and '+prop_attr
								else:
									attr_text += ', '+prop_attr
								added.add(prop_attr)
						else:
							attr_text = prop_split[1]

						if pluralize.singular_noun(term) is not False:
							prop_aggregated.append(attr_text+' '+term)

						elif prop_ordered.count(prop)>1 and pluralize.singular_noun(term) is False:
							prop_aggregated.append(attr_text+' '+pluralize.plural_noun(term))

						else:
							prop_aggregated.append(prop_split[0]+' '+attr_text+' '+term)	


					prop_to_add = []
					for prop in prop_aggregated:
						if len(prop_to_add)>=num_prop:
							break
						if prop not in enhanced_cap[num_prop][img_id]:
							prop_to_add.append(prop)
					if len(prop_to_add)<1:
						continue 

					text = ''
					for i,prop in enumerate(prop_to_add): 
						if i == 0:
							text += prop
						elif i == len(prop_to_add)-1:
							if len(prop_to_add) == 2:
								text += ' and '+prop
							else:
								text += ', and '+prop
						else:
							text += ', '+prop

					text = ' with '+text
					enhanced_cap[num_prop][img_id] += text

for num_prop in num_props:
	cap_out = []
	for entry in enhanced_cap[num_prop].keys():
		cap_out.append({'image_id':str(entry),'caption':enhanced_cap[num_prop][entry]}) 


	with open('results/captions_out_'+dataset+'_'+model+'_TROPE_'+str(num_prop)+'parts_frac'+str(frac_thres)+'_new.json', 'w') as fh:
		json.dump(cap_out,fh)	
		fh.close()
