#!/bin/bash
nvidia-docker run -it \
  -p 8888:8888 \
  -v $PWD/scripts:/notebooks \
  -v $PWD/docker_data:/data \
  -v $PWD/deepcell:/usr/local/lib/python3.6/dist-packages/deepcell/ \
  noahgreenwald/deepcell-tf:dev
