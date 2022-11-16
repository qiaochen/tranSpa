# coding:utf-8

from distutils.core import setup

setup(name='transpa',
      version='0.0.1',
      description='Translation-based imputation and cell type deconvolution',
      author='Chen Qiao',
      author_email='cqiao@connect.hku.hk',
      url='https://github.com/qiaochen/tranSpa',
      packages=['transpa'],
    #   entry_points = {
    #         "console_scripts": ['veloproj = veloproj.veloproj:main']
    #   },
      install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'tqdm',
        'anndata>=0.7.4',
        'scanpy>=1.5.1',
        'torch>=1.7',
        'scikit-learn',
      ]
    )