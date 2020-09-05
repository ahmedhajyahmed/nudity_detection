"""restart package

This script allows the user to uninstall nudity_detection package,
remove dist, build and nudity_detection.egg-info folders and
re-install the package.

Usage:
    python restart_package.py

Author:
    Ahmed Haj Yahme (hajyahmedahmed@gmail.com)
"""
import os
import shutil
os.system('pip uninstall nudity_detection')
shutil.rmtree('./dist', ignore_errors=True)
shutil.rmtree('./build', ignore_errors=True)
shutil.rmtree('./nudity_detection.egg-info', ignore_errors=True)
os.system('python setup.py sdist bdist_wheel')
os.system('pip install dist/nudity_detection-0.1-py3-none-any.whl')
