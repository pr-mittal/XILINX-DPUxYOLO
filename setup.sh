#https://xilinx.github.io/Vitis-AI/docs/install/install.html#build-docker-from-scripts

#NVIDIA
#PRE INSTALLATION
lspci | grep -i nvidia
uname -m && cat /etc/*release
gcc --version
uname -r
sudo apt-get install linux-headers-$(uname -r)
#INSTALLATION
sudo apt-key del 7fa2af80
wget https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
#ubuntu1604/x86_64
#ubuntu1804/cross-linux-sbsa
#ubuntu1804/ppc64el
#ubuntu1804/sbsa
#ubuntu1804/x86_64
#ubuntu2004/cross-linux-sbsa
#ubuntu2004/sbsa
#ubuntu2004/x86_64
#ubuntu2204/sbsa
#ubuntu2204/x86_64
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda 
sudo apt-get install nvidia-gds
sudo reboot
#POST INSTALLATION
vim ~/.bashrc
export PATH=/usr/local/cuda-12.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
#apt purge nvidia* libnvidia*
#apt install nvidia-driver-xxx
#apt install nvidia-container-toolkit
nvidia-smi
#DOCKER
for pkg in docker.io docker-doc docker-compose podman-docker containerd runc; do sudo apt-get remove $pkg; done
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
docker --version
sudo groupadd docker
sudo usermod -aG docker $USER
sudo gpasswd -a $USER docker
newgrp docker
docker run hello-world
#DOCKER BUILD
sudo apt-get install -y nvidia-docker2
git clone https://github.com/Xilinx/Vitis-AI
cd Vitis-AI/docker
./docker_build.sh -t gpu -f pytorch
docker run --gpus all nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04 nvidia-smi

################### OR ################################################
#python version - 3.7.12
#use pyenv
#local install 
https://github.com/Xilinx/Vitis-AI/tree/master/src/vai_quantizer/vai_q_pytorch
#pip install torch==1.7.0+cu110 torchvision==0.8.0+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install protobuf==3.20.*
#conda
Please try the following steps:

    create a new clean conda env and activate it
    install pytorch 1.6 and pytorch_nndct following the instructions in https://github.com/Xilinx/Vitis-AI/tree/master/src/vai_quantizer/vai_q_pytorch.
    get XIR package at https://www.xilinx.com/bin/public/openDownload?filename=conda-channel_1.4.914-01.tar.gz:

   cd /tmp
   wget -O conda-channel.tar.gz https://www.xilinx.com/bin/public/openDownload?filename=conda-channel_1.4.914-01.tar.gz
   tar zxf conda-channel.tar.gz 

    install XIR into your conda environment with:
    conda install xir -c file:///tmp/conda-channel
    conda install xir -c /home/ubuntu/Downloads/conda-channel

#download dataset
https://drive.google.com/file/d/1ceQ5y_rCReSZ26HzzCf2muDNbovjyl5k/view?usp=share_link
