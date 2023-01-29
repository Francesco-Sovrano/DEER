echo 'Create a virtual environment'
python3.7 -m venv .env

echo 'Activate the virtual environment'
source .env/bin/activate
# echo 'Update the virtual environment'
pip install -U pip setuptools wheel psutil Cython

echo 'Installing env CarController..'
pip install -r environments/bot_controller/requirements.txt

# echo 'Installing env SpecialAtari..'
# pip install -r environments/special_atari/requirements.txt
# pip install gym[atari]==0.14.0
# mkdir Atari-ROM
# cd Atari-ROM
# wget http://www.atarimania.com/roms/Roms.rar
# unrar x Roms.rar
# unzip ROMS.zip
# python -m atari_py.import_roms ROMS
# cd ..

# echo 'Installing env Shepherd..'
# pip install -r environments/shepherd/requirements.txt

# echo 'Installing env GFootball..'
# pip install -r environments/gfootball/requirements.txt

# echo 'Installing env PettingZoo..'
# pip install -r environments/petting_zoo/requirements.txt

echo 'Installing env PRIMAL..'
pip install -r environments/primal/requirements.txt
cd environments/primal/od_mstar3
python3 setup.py build_ext --inplace
rm -r build
cd ../../..
cd environments/primal/astarlib3
python3 setup.py build_ext --inplace
rm -r build
cd ../../..

echo 'Installing DEER..'
pip install -e ./package # cmake is needed
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.1+cpu.html

# echo 'Fixing environments rendering'
# pip install pyglet==1.5.11 # fix for rendering environments
# echo 'Fixing protobuf'
# pip install protobuf==3.20.* # https://github.com/protocolbuffers/protobuf/issues/10051
