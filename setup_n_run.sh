#!/bin/bash

MY_DIR="`dirname $(readlink -f \"$0\")`"
EXPERIMENT_ID=$3

# pyclean
cd $MY_DIR
# (find ./ -name __pycache__ -type d | xargs rm -r) && (find ./ -name *.pyc -type f | xargs rm -r)

if [ ! -d ".env" ]; then
	echo 'Create a virtual environment'
	python3.7 -m venv .env

	echo 'Activate the virtual environment'
	. .env/bin/activate
	# echo 'Update the virtual environment'
	pip install -U pip setuptools wheel psutil Cython

	echo 'Installing DEER..'
	pip install -e ./package # cmake is needed
	pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.1+cpu.html

	echo 'Installing Fuzzing libs..'
	cd environment/fuzzing
	pip install -r requirements.txt
	cd cpp_mods
	make clean
	make all
	cd ../../..
else
	. .env/bin/activate	
fi

if [ ! -d "result" ]; then
	mkdir result
fi
cd ./result

if [ ! -d "$EXPERIMENT_ID" ]; then
  mkdir $EXPERIMENT_ID
fi
cd $EXPERIMENT_ID

# python3 $MY_DIR/$EXPERIMENT_ID/train.py
# ulimit -n 100000
python3 $MY_DIR/commandline_training_script.py $* &>> out.log &
disown
exit
