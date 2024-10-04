# TRaining-Free Object-Part Enhancement (TROPE)

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Caption enhancement algorithm described in the paper "TROPE: TRaining-Free Object-Part Enhancement for Seamlessly Improving Fine-Grained Zero-Shot Image Captioning" (EMNLP 2024 Findings).

arXiv: https://www.arxiv.org/abs/2409.19960

ACL Anthology: arriving soon

### Requirements
You can run requirements/install.sh to quickly install all the requirements in an Anaconda environment. The requirements are:
- python 3
- inflect>=7.4.0
- more_itertools>=10.5.0
- spacy
- numpy

### Usage

./structured_cap.sh provides working a working ablation study which passes arguments to the trope/generate_proposals.py script. The script can be run directly using the following example: python3 trope/generate_proposals.py CUB 0.5 oscar

#### Repository Data
Example outputs for TROPE can be found in the results directory. Additional documentation will be added soon.

#### Evaluation
Output captions found in the results directory and reference captions are compatible can be fed directly into standard evaluation protocols (CIDEr, METEOR, and SPICE: https://github.com/tylin/coco-caption | SMURF: https://github.com/JoshuaFeinglass/SMURF).

### Author/Maintainer:
Joshua Feinglass (https://scholar.google.com/citations?user=V2h3z7oAAAAJ&hl=en)

If you find this repo useful, please cite:
```
@inproceedings{feinglass2024trope,
  title={TROPE: TRaining-Free Object-Part Enhancement for Seamlessly Improving Fine-Grained Zero-Shot Image Captioning},
  author={Joshua Feinglass and Yezhou Yang},
  booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP) Findings},
  year={2024},
  url={https://www.arxiv.org/pdf/2409.19960}
}
```
