echo 'Create a virtual environment'
python3.7 -m venv .env

echo 'Activate the virtual environment'
source .env/bin/activate
# echo 'Update the virtual environment'
pip install -U pip setuptools wheel psutil Cython

echo "GridDrive and GraphDrive"
pip install -r environment/requirements.txt

echo 'Installing DEER..'
pip install -e ./package # cmake is needed
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.1+cpu.html

# echo 'Fixing environment rendering'
# pip install pyglet==1.5.11 # fix for rendering environment
# echo 'Fixing protobuf'
# pip install protobuf==3.20.* # https://github.com/protocolbuffers/protobuf/issues/10051
