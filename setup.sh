echo 'Create a virtual environment'
python3.7 -m venv .env

echo 'Activate the virtual environment'
source .env/bin/activate
# echo 'Update the virtual environment'
pip install -U pip setuptools wheel psutil Cython

echo 'Installing DEER..'
pip install -e ./package # cmake is needed
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.1+cpu.html

# echo 'Fixing environment rendering'
# pip install pyglet==1.5.11 # fix for rendering environment
# echo 'Fixing protobuf'
# pip install protobuf==3.20.* # https://github.com/protocolbuffers/protobuf/issues/10051

echo "GridDrive and GraphDrive"
pip install -r environment/requirements.txt

echo "MuJoCo"
# Set up Mujoco path variable (usually it's ~/.mujoco)
MUJOCO_PATH=$HOME/.mujoco
MUJOCO_VERSION="mujoco-3.0.0" # Use the version compatible with your gym version
MUJOCO_ZIP_FILE="${MUJOCO_VERSION}_linux.tar.gz"
MUJOCO_KEY_PATH=$HOME # Or wherever you want to place your MuJoCo license key

# Create the .mujoco directory if it doesn't exist
mkdir -p $MUJOCO_PATH

# Download the MuJoCo binaries - You will need to replace this with the actual download command 
# as downloading MuJoCo typically requires accepting terms of service.
echo "Downloading MuJoCo binaries..."
wget -O $MUJOCO_PATH/$MUJOCO_ZIP_FILE https://github.com/google-deepmind/mujoco/archive/refs/tags/3.0.0.tar.gz

# Unzip the downloaded binaries
tar -xf $MUJOCO_PATH/$MUJOCO_ZIP_FILE -C $MUJOCO_PATH

# # Copy the license key file to the MuJoCo directory
# # Please replace 'mjkey.txt' with the actual name of your MuJoCo license key file
# cp $MUJOCO_KEY_PATH/mjkey.txt $MUJOCO_PATH/${MUJOCO_VERSION}/

# Append the MuJoCo directory to the LD_LIBRARY_PATH environment variable
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$MUJOCO_PATH/${MUJOCO_VERSION}/bin" >> $HOME/.bashrc
source $HOME/.bashrc

# Install the mujoco-py Python package
pip install mujoco gym==0.25.*
pip install "cython<3"
python -c 'import mujoco'

# The above steps assume you have a MuJoCo license key.
# As of my last update, MuJoCo is open source and you may not need a license key.
# Please verify the current situation as this may have changed.

echo "Installation complete. Please verify that MuJoCo is working correctly."

pip install numpy==1.21.6