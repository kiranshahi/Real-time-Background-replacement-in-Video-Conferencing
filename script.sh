cd $HOME/dissertation
source tf/bin/activate
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/hdd/cudnn-8.2.1-cuda11.3_0/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/hdd/lib
export CUDA_VISIBLE_DEVICES=4
nohup time python3 model.py &