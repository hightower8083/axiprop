# Copyright 2020, Igor Andriyash
# Authors: Igor Andriyash
# License: BSD-3-Clause
from setuptools import setup, find_packages
import axiprop

# Obtain the long description from README.md
# If possible, use pypandoc to convert the README from Markdown
# to reStructuredText, as this is the only supported format on PyPI
with open("./README.md", encoding="utf-8") as f:
     long_description = f.read()

# Get the package requirements from the requirements.txt file
with open('requirements.txt') as f:
    install_requires = [ line.strip('\n') for line in f.readlines() ]

setup(
    name='axiprop',
    python_requires='>=3.6',
    version=axiprop.__version__,
    description="simple-to-use optical propagation tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Igor Andriyash',
    maintainer='Igor Andriyash',
    maintainer_email='igor.andriyash@gmail.com',
    license='BSD-3-Clause',
    license_files=["LICENSE",],
    packages=find_packages('.'),
    package_data={"": ['*']},
    tests_require=[],
    cmdclass={},
    install_requires=install_requires,
    include_package_data=True,
    platforms='any',
    url='https://github.com/hightower8083/axiprop',
    classifiers=[
        'Programming Language :: Python',
        'Natural Language :: English',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Physics',
        'Programming Language :: Python :: 3.6',
        ],
    zip_safe=False,
    )
