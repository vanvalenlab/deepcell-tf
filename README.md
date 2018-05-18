# deepcell-tf


### NVIDIA GPU command
nvidia-smi

### Docker Commands

##### Python 2.7 deepcell
sudo nvidia-docker run -i -t \
-v /home/vanvalen/deepcell-tf/deepcell:/usr/local/lib/python2.7/dist-packages/deepcell/ \
-v /home/vanvalen/deepcell-tf/deepcell_scripts:/deepcell-tf/deepcell_scripts \
-v /home/vanvalen/data/old_training_data:/data \
nvcr.io/vvlab/deepcell:0.1


##### Python 3 deepcell
sudo NV_GPU='1,2' nvidia-docker run -i -t \
-v /home/vanvalen/deepcell-tf/deepcell_tf/deepcell:/usr/local/lib/python3.5/dist-packages/deepcell/ \
-v /home/vanvalen/deepcell-tf/deepcell_scripts:/deepcell-tf/deepcell_scripts \
-v /home/vanvalen/data/old_training_data:/data \
dylan/deepcell-tf:0.1dev
