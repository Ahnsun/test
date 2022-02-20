#!/usr/bin/env bash

cd pysot/utils/
python setup.py clean
python setup.py build_ext --inplace
cd ../../
