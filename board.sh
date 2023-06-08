#host
dmesg | grep tty
sudo putty /dev/ttyUSB1 -serial -sercfg 115200,8,n,1,N
#remote
# ubuntu , password:ubuntu
#ubuntu 20.x
sudo snap install xlnx-config --classic --channel=1.x
sudo xlnx-config --xmutil bootfw_update -i <path to boot.bin>
#ubuntu 22.x
sudo snap install xlnx-config --classic --channel=2.x
sudo xlnx-config.sysinit

#pynq
mkdir /home/ubuntu/git
cd /home/ubuntu/git
git clone https://github.com/Xilinx/Kria-PYNQ.git
cd Kria-PYNQ/
sudo bash install.sh -b KV260 
ifconfig
#http://192.168.18.226:9090/tree

mkdir /home/ubuntu/git
cd /home/ubuntu/git
git clone https://github.com/pr-mittal/dac_sdc_2023.git -b deploy --single-branch --depth 1 /home/ubuntu/git/fpga_starter_2023
ln -s /home/ubuntu/git/fpga_starter_2023 /home/root/jupyter_notebooks
#pull the changes in deploy branch
#git pull origin deploy
#git push origin localbranchname:remotebranchname
