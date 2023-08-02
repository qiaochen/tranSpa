[![PyPi][badge-pypi]][link-pypi]
[![DOC][badge-doc]][link-doc]
[![CI][badge-ci]][link-ci]

[badge-pypi]: https://badge.fury.io/py/transpa.svg
[link-pypi]: https://pypi.org/project/transpa/
[badge-doc]: https://readthedocs.org/projects/transpa/badge/?version=latest
[link-doc]: https://transpa.readthedocs.io/en/latest/
[badge-ci]: https://api.travis-ci.com/qiaochen/tranSpa.svg?branch=main
[link-ci]: https://app.travis-ci.com/github/qiaochen/tranSpa

# TranSpa
This tool implements Spatially-Regularized Translation for imputing spatial transcriptomics (TransImpSpa), and translation based cell type deconvolution (TransDeconv). Experiments reported in the manuscript are displayed in jupyter notebooks under [notebooks](https://github.com/qiaochen/tranSpa/tree/analysis/notebooks) folder.

Three demo notebooks are also available under the [demo](https://github.com/qiaochen/tranSpa/tree/main/demo) folder.

- [Different configurations of TransImp applied to SeqFISH dataset dataset](https://github.com/qiaochen/tranSpa/blob/main/demo/seqfish.ipynb)
- [Exploration for unprobed genes with SeqFISH ST dataset](https://github.com/qiaochen/tranSpa/blob/main/demo/seqfish_unprobed_genes.ipynb)
[Cell type deconvolution with TransDeconv](https://github.com/qiaochen/tranSpa/blob/main/demo/transDeconv.ipynb)

## Installation

TransImp is available through PyPI. To install, type the following command line and add -U for updates:

```
pip install -U transpa
```

Or, download the project and under project root `tranSpa/`

```
pip install .
```

## Data
Data used and generated in the notebooks can be downloaded from [Zenodo](https://zenodo.org/record/7556184#.Y8tfmXZBxD8)
To run the notebooks, please extract the root folders as sibling folders to `tranSpa`, which should be `../data` and `../output` relative to the `README.md` file

## Documentation
Please visit [TransImp website](https://transpa.readthedocs.io/en/latest/) for more details.




