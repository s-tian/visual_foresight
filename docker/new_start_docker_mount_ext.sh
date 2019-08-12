# instructions for running this the first time:
# pull the docker image from docker hub: run ` docker pull febert/recplan:latest `
docker run  -v $VMPC_EXP/:/workspace/experiments \
                   -v $VMPC_DATA/:/workspace/data/vmpc_data  \
                   -v $NAS_CODE/:/workspace/code \
                   -v /raid/:/raid \
                   -v /:/parent \
                   -v /nfs:/nfs \
                   --name=${DOCKER_NAME}  \
-e VMPC_EXP=/workspace/experiments \
-e VMPC_DATA=/workspace/data/vmpc_data \
-e RAY_RESULTS=/parent/nfs/kun1/users/febert/data/ray_results \
-e NAS_CODE=/workspace/code \
-t -d \
--shm-size 8G \
febert/vmpc_image:latest \
/bin/bash
docker exec ${DOCKER_NAME} /bin/bash

