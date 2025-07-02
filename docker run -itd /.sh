docker run -itd \
           --net=host \
           --device=/dev/dri \
           --memory="24G" \
           --name=$CONTAINER_NAME \
           --shm-size="16g" \
           --entrypoint /bin/bash \
           $DOCKER_IMAGE