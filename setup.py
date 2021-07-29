from setuptools import setup, find_packages

setup(
  name = 'multistream-transformers',
  packages = find_packages(),
  version = '0.0.1',
  license='MIT',
  description = 'Multistream Transformers - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/multistream-transformers',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'efficient attention'
  ],
  install_requires=[
    'einops>=0.3',
    'torch>=1.6'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)