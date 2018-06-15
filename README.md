# deepcell-tf


### NVIDIA GPU activity monitor
`nvidia-smi`

### Docker Commands

##### Python 2.7 deepcell
```bash
nvidia-docker run -i -t \
-v $HOME/deepcell-tf/deepcell:/usr/local/lib/python2.7/dist-packages/deepcell/ \
-v $HOME/deepcell-tf/scripts:/deepcell-tf/scripts \
-v /home/vanvalen/data/training_data:/data \
nvcr.io/vvlab/deepcell:0.1
```

##### Python 3 deepcell
```bash
NV_GPU='1,2' nvidia-docker run -i -t \
-v $HOME/deepcell-tf/deepcell:/usr/local/lib/python3.5/dist-packages/deepcell/ \
-v $HOME/deepcell-tf/scripts:/deepcell-tf/scripts \
-v /home/vanvalen/data/training_data:/data \
nvcr.io/vvlab/deepcell:0.1
```

#### Python3 jupyter notebook
```bash
NV_GPU='1,2' nvidia-docker run -i -t -p 80:8888 \
-v $HOME/deepcell-tf/deepcell:/usr/local/lib/python3.5/dist-packages/deepcell/ \
-v $HOME/deepcell-tf/scripts:/deepcell-tf/scripts \
-v /home/vanvalen/data/training_data:/data \
--entrypoint /usr/local/bin/jupyter \
nvcr.io/vvlab/deepcell:0.1 \
notebook --allow-root --ip=0.0.0.0
```
