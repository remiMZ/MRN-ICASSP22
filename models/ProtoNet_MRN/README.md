# Prototypical network + MRN

#### 5-way-1-shot

- Training with single GPU:

CUDA_VISIBLE_DEVICES=0 python main.py --data-folder "" --backbone "conv4" --num-shots 1 --train-tasks 60000 --test-data "miniimagenet" --regular-type "MLP" 

- Training with multiple GPUs by torch.nn.DataParallel:

CUDA_VISIBLE_DEVICES=0,1 python main.py --data-folder "" --backbone "conv4" --num-shots 1 --train-tasks 60000 --test-data "miniimagenet" --regular-type "MLP" --multi-gpu


#### 5-way-5-shot

- Training with single GPU:

CUDA_VISIBLE_DEVICES=0 python main.py --data-folder "" --backbone "conv4" --num-shots 5 --train-tasks 40000 --test-data "miniimagenet" --regular-type "MLP"

- Training with multiple GPUs by torch.nn.DataParallel:

CUDA_VISIBLE_DEVICES=0,1 python main.py --data-folder "" --backbone "conv4" --num-shots 5 --train-tasks 40000 --test-data "miniimagenet" --regular-type "MLP" --multi-gpu