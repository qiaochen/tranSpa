#!/bin/bash

python preprocess_osmFISH_AllenVISp.py
python preprocess_seqFISH_singlecell.py
python preprocess_starmap_AllenVISp.py
python preprocess_merfish_moffit.py

python preprocess_intestine.py
python preprocess_breastcancer.py
python preprocess_melanoma.py
python preprocess_mouseliver.py