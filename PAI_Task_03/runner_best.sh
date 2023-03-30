docker build --tag task3 . > /dev/null
x=0
while true
do
    echo "Started $x iteration"
    BASH_VAR=$(docker run  --mount type=bind,source=/home/simone/docker_tmp,target=/root --rm -v "$( cd "$( dirname "$0" )" && pwd )":/results task3)
    VAR=${BASH_VAR#*Your average regret is }
    VAR=${VAR:0:10}
    echo $VAR 
    mv results_check.byte $VAR.byte
    x=$(( $x + 1 ))
done

