|Docs| |Build Status|

.. .. |PyPI| image:: https://img.shields.io/pypi/v/transp.svg
..     :target: https://pypi.org/project/transpa/
.. |Docs| image:: https://readthedocs.org/projects/transpa/badge/?version=latest
   :target: https://transpa.readthedocs.io/en/latest/
.. |Build Status| image:: https://api.travis-ci.com/qiaochen/tranSpa.svg?branch=main
   :target: https://app.travis-ci.com/github/qiaochen/tranSpa
====
Home
====


About TransImp
===============

TransImp (Translation-based Spatial Transcriptomics Imputation toolkit) is a computational framework and toolbox for reference-based imputation of spatial transcriptomic dataset.

.. .. image:: x
..    :width: 900px
..    :align: center
 

.. .. image:: https://github.com/StatBiomed/SpatialDM/blob/main/docs/.figs/runtime_aug16-1.png?raw=true
..    :width: 600px
..    :align: center

TransImp features in two novel functions: \
 1) Uncertainty score estimation for each imputed gene, allowing the identification of relible predictions; \
 2) Spatial regularisation for explicitily control the degree of spatial pattern preservation for imputation modeling.

Please refer to our demo notebooks for details:

* `Different configurations of TransImp applied to SeqFISH dataset dataset`_.

* `Exploration for unprobed genes with SeqFISH ST dataset`_.

.. _Different configurations of TransImp applied to SeqFISH dataset dataset: seqfish.ipynb

.. _Exploration for unprobed genes with SeqFISH ST dataset: seqfish_unprobed_genes.ipynb



References
==========
Manuscript of TransImp with more details is available on bioRxiv_ now and is currently under review.

.. _bioRxiv: https://www.biorxiv.org/content/10.1101/2023.01.20.524992v1



.. toctree::
   :caption: Main
   :maxdepth: 1
   :hidden:
   
   install
   api
   release

.. toctree::
   :caption: Examples
   :maxdepth: 1
   :hidden:

   seqfish
   seqfish_unprobed_genes