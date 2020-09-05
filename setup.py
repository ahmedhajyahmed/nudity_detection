from setuptools import setup, find_packages

with open('requirements.txt') as requirements_file:
    requirements = requirements_file.read().splitlines()

setup_args = dict(
    name='nudity_detection',
    version='0.1',
    description='package containing a deep learning model for detecting nudity in a picture',
    long_description='package containing a deep learning model for detecting nudity in a picture',
    classifiers=[
            'Programming Language :: Python :: 3.6.6',
            'Topic :: image Processing :: nudity detection',
          ],
    packages=find_packages(),
    author='Ahmed Haj Yahmed',
    author_email='hajyahmedahmed@gmail.com',
    keywords=['Nudity detection', 'Nudity classification', 'Image classification'],
)

install_requires = requirements

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)