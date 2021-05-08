name: stack
channels:
  - defaults
dependencies:
  - python=3.6  # Google Colab is still on Python 3.6
  - cudatoolkit=10.1
  - cudnn=7.6
  - pip
  - pip:
    - pip-tools
