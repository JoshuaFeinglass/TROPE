# TRaining-Free Object-Part Enhancement (TROPE)

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Post-processing caption enhancement algorithm described in the paper "TROPE: TRaining-Free Object-Part Enhancement for Seamlessly Improving Fine-Grained Zero-Shot Image Captioning" (EMNLP 2024 Findings). Supplemental part information from an object detector is added to a base caption to improve fine-grained detail.

arXiv: https://www.arxiv.org/abs/2409.19960

ACL Anthology: https://aclanthology.org/2024.findings-emnlp.207/

### Requirements
You can run requirements/install.sh to quickly install all the requirements in an Anaconda environment. The requirements are:
- python 3
- inflect>=7.4.0
- more_itertools>=10.5.0
- spacy
- numpy

### Usage
./run_TROPE.sh provides working a working ablation study which passes arguments to the trope/trope_main.py script. The script can be run directly using the following example: python3 trope/trope_main.py CUB 0.5 oscar

#### Repository Data
Example outputs for TROPE can be found in the results/ directory. Base captions from the VINVL+Oscar captioning pipeline for 4 fine-grained datasets (CUB, FLO, UCM, and SC) are in the base_captions/ directory. The detector_info/ directory contains the category set mappings for VINVL as well as the test set ids, attribute labels, object labels, bounding boxes, and confidence scores needed for caption enhancement of the 4 fine-grained datasets with the TROPE algorithm.

#### Evaluation
Output captions found in the results directory and reference captions are compatible and can be fed directly into standard evaluation protocols (CIDEr, METEOR, and SPICE: https://github.com/tylin/coco-caption | SMURF: https://github.com/JoshuaFeinglass/SMURF).

### Author/Maintainer:
Joshua Feinglass (https://scholar.google.com/citations?user=V2h3z7oAAAAJ&hl=en)

If you find this repo useful, please cite:
```
@inproceedings{feinglass2024trope,
  title={TROPE: TRaining-Free Object-Part Enhancement for Seamlessly Improving Fine-Grained Zero-Shot Image Captioning},
  author={Joshua Feinglass and Yezhou Yang},
  booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP) Findings},
  year={2024},
  url={https://aclanthology.org/2024.findings-emnlp.207/}
}
```
