#! /bin/bash

# Check if IP address is provided as first argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <IP_ADDRESS>"
    echo "Example: $0 3.136.54.71"
    exit 1
fi

IP_ADDRESS=$1

### For VWA:
bash reset_reddit.sh
bash reset_shopping.sh
curl -X POST "http://${IP_ADDRESS}:9980/index.php?page=reset" -d "token=4b61655535e7ed388f0d40a93600254c"

### For WebArena:
docker stop shopping_admin gitlab
docker remove shopping_admin gitlab
docker run --name shopping_admin -p 7780:80 -d shopping_admin_final_0719
docker run --name gitlab -d -p 8023:8023 gitlab-populated-final-port8023 /opt/gitlab/embedded/bin/runsvdir-start