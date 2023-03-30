docker build --tag task3 .
docker run  --mount type=bind,source=/home/simone/docker_tmp,target=/root --rm -v "$( cd "$( dirname "$0" )" && pwd )":/results task3