#!/bin/bash
nvidia-smi
nnode=$1
node_rank=$2
nproc_per_node=$3
master_addr=$4
master_port=$(( ( RANDOM % 3000 )  + 27000 ))
echo "nnode = $nnode"
echo "node_rank = $node_rank"
echo "nproc_per_node = $nproc_per_node"
echo "master_addr = $master_addr"
echo "master_port = $master_port"
shift
shift
shift
shift
shift
echo "$@"
python -u -m torch.distributed.launch --nnode=$nnode --node_rank=$node_rank --nproc_per_node=$nproc_per_node --master_addr=$master_addr --master_port=$master_port train.py "$@"

