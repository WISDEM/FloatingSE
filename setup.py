#!/usr/bin/env python
# encoding: utf-8

from setuptools import setup

setup(
    name='FloatingSE',
    version='0.5',
    description='Floating Substructure Systems Engineering Model',
    author='Garrett Barter',
    author_email='garrett.barter@nrel.gov',
    install_requires=['openmdao>=1.6', 'commonse', 'pyframe3dd>=1.1'],
    package_dir={'': 'src'},
    packages=['floatingse'],
    license='Apache License, Version 2.0',
    zip_safe=False
)
