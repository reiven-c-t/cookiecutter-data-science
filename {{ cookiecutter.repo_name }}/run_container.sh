#!/usr/bin/env bash

# fix -p external1 as extanal port 1 and internal1 as internal port 1
# fix --name container name you want
# fix --rm image:tag

nvidia-docker run -it -v `pwd`:/data -p extermal1:internal1 -p external2:internal2 --name=nmc_container --rm image:tag /bin/bash