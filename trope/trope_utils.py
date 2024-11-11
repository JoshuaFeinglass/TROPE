import csv
import json
from collections import namedtuple
import spacy
import numpy as np
import inflect
pluralize = inflect.engine()
adj_list = ['long','short','yellow','orange','green','red','rectangular','pink','blue','blond','blank','black and white','black','white']
Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
nlp = spacy.load("en_core_web_md")

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

def extract_key_objects(base_cap, bboxes, labs):
	toks = nlp(base_cap)
	stem_labs = set(entry.lower() for entry in labs)
	key_objs = []
	caption_objs = []
	obj_locs = []
	check_plural = []
	for i,tok in enumerate(toks):
		tok_lowered = tok.text.lower()
		caption_objs.append(tok_lowered)
		check_plural.append(pluralize.singular_noun(tok_lowered))
		if check_plural[-1] is False:
			stem_tok = tok_lowered
		else:
			stem_tok = check_plural[-1]
		if stem_tok in stem_labs and tok.pos_ == 'NOUN' and toks[i-1].pos_ != 'NOUN' and toks[i+1].pos_ != 'NOUN' and toks[i+1].text.lower() != 'of':
			key_objs.append(stem_tok)
			obj_locs.append(i)
			caption_objs.append(stem_tok)

	obj_boxes = {}
	obj_areas = []
	for j,obj in enumerate(key_objs):	
		for i,entry in enumerate(labs):
			if entry == obj:
				if check_plural[obj_locs[j]] is False and obj not in obj_boxes.keys():
					obj_boxes[obj] = bboxes[i]
				elif check_plural[obj_locs[j]] is False and obj in obj_boxes.keys():
					continue
				elif check_plural[obj_locs[j]] is not False and obj not in obj_boxes.keys():
					obj_boxes[obj] = None
					obj_boxes[obj] = combine_rects(obj_boxes[obj],bboxes[i])
				else:
					obj_boxes[obj] = combine_rects(obj_boxes[obj],bboxes[i])

		obj_areas.append(area(obj_boxes[obj]))

	key_objs = [key_objs[i] for i in np.argsort(obj_areas)]
	obj_locs = [obj_locs[i] for i in np.argsort(obj_areas)]

	return key_objs, obj_locs, caption_objs, obj_boxes, toks

def generate_proposals(key_objs, caption_objs, obj_boxes, bboxes, labs, scores, attr_labs, frac_thres):
	part_props = {}
	added_props = set()
	for obj in key_objs:
		part_props[obj] = []
		to_add = set()
		for ind,bbox in enumerate(bboxes):
			if labs[ind] in caption_objs or labs[ind] in added_props:
				continue
			
			area_bbox = area(bbox)
			frac = inter_area(obj_boxes[obj],bbox)/area_bbox

			if frac >= frac_thres and labs[ind] not in adj_list:

				attr_to_add = attr_labs[ind] 
				
				article = 'a'
				if attr_to_add[0] in ['a','e','i','o','u']:
					article = 'an'

				to_add.add(labs[ind])

				part_tup = (article+' '+attr_to_add+' '+labs[ind],frac+float(scores[ind]))

				part_props[obj].append(part_tup)

		added_props = added_props.union(to_add)

	return part_props

def aggregate_and_add_to_caption(enhanced_cap, key_objs, key_obj_locs, part_props, toks, num_prop):
	prop_added = False
	for i,tok in enumerate(toks):
		if prop_added == True and (tok.text == 'with' or tok.text == 'has' or tok.text == 'have'):
			prop_added = False
			enhanced_cap += ' in addition to'
			continue 

		prop_added = False

		if tok.text == '\'s' or toks[i-1].text == '\'s':
			continue
		
		if i != 0 and tok.text not in  ['.',',',':',';'] :
			enhanced_cap += ' '

		enhanced_cap += tok.text.lower()
		if i<(len(toks)-1) and toks[i+1].text == '\'s':
			enhanced_cap += toks[i+1].text	
			enhanced_cap += ' '+toks[i+2].text	

		
		if len(key_obj_locs)>0 and key_obj_locs[0] == i:
			del key_obj_locs[0]
			obj = key_objs.pop(0)
			part_props[obj] = [prop for prop in part_props[obj] if prop is not None]
			if len(part_props[obj])>0:
				prop_added = True
				prop_order = np.flipud(np.argsort([prop[1] for prop in part_props[obj]]))
				prop_ordered = [part_props[obj][entry][0] for i,entry in enumerate(prop_order)]

				
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
					if prop not in enhanced_cap:
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
				enhanced_cap += text
	return enhanced_cap	
