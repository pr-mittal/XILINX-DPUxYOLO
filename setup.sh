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
sudo docker run hello-world
docker --version
#DOCKER BUILD
git clone https://github.com/Xilinx/Vitis-AI
cd Vitis-AI/docker
./docker_build.sh -t gpu -f pytorch
docker run --gpus all nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04 nvidia-smi
