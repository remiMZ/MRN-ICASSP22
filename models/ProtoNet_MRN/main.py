import os
import time
import json
import torch
import random
import torch.nn as nn
from tqdm import tqdm
from pandas import DataFrame
from utils import get_accuracy, fix_nn 
from model import PrototypicalNetwork, RegularNetwork
import sys
sys.path.append("..")
sys.path.append("../..")

from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.prototype import get_prototypes, prototypical_loss
from global_utils import get_dataset, Averager, Mean_confidence_interval, get_outputs_c 

def save_model(model, args, tag):
    model_path = os.path.join(args.record_folder, ('_'.join([args.model_name, args.test_data, args.backbone, tag]) + '.pt'))
    if args.multi_gpu:
        model = model.module
    with open(model_path, 'wb') as f:
        torch.save(model.state_dict(), f)

def save_checkpoint(args, model, train_log, optimizer, global_task_count, tag):
    if args.multi_gpu:
        model = model.module
    state = {
        'args': args,
        'model': model.state_dict(),
        'train_log': train_log,
        'val_acc': train_log['max_acc'],
        'optimizer': optimizer.state_dict(),
        'global_task_count': global_task_count
    }
    checkpoint_path = os.path.join(args.record_folder, ('_'.join([args.model_name, args.train_data, args.test_data, args.backbone, tag]) + '_checkpoint.pt.tar'))
    with open(checkpoint_path, 'wb') as f:
        torch.save(state, f)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Domain Generalization Meta-Learning (MRN)')
    parser.add_argument('--model-name', type=str, default='Proto_MRN', help='Name of the model.')
    
    parser.add_argument('--data-folder', type=str, default='./dataset',
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--train-data', type=str, default= 'cub', help='Name of train-data.')
    parser.add_argument('--test-data', type=str, default= 'cub', help='Name of test-data.')
    parser.add_argument('--backbone', type=str, default='conv4',
        help='The type of model backbone.')
    
    parser.add_argument('--num-shots', type=int, default=5,
        help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-ways', type=int, default=5, 
        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--test-shots', type=int, default=15,
        help='Number of examples per class (k in "k-shot", default: 16).')
    
    parser.add_argument('--batch-tasks', type=int, default=4,
        help='Number of tasks in a mini-batch of tasks (default: 4).')
    parser.add_argument('--train-tasks', type=int, default=40000,
        help='Number of tasks in the training phase (default: 40000).')
    parser.add_argument('--val-tasks', type=int, default=600,
        help='Number of tasks the model network is validated over (default: 600). ')
    parser.add_argument('--test-tasks', type=int, default=1000,
        help='Number of tasks the model network is tested over (default: 1000). The final results will be the average of these batches.')
    parser.add_argument('--validation-tasks', type=int, default=1000,
        help='Number of tasks for each validation (default: 1000).')
    
    parser.add_argument('--omega', type=float, default=1e-4,
        help='Initial learning rate of regularnetwork (default: 1e-4).')
    parser.add_argument('--lr', type=float, default=0.001,
        help='Initial learning rate of model (default: 0.001).')
    parser.add_argument('--schedule', type=int, nargs='+', default=[15000, 30000, 45000, 60000], 
        help='Decrease learning rate at these number of tasks.')
    parser.add_argument('--gamma', type=float, default=0.1,
        help='Learning rate decreasing ratio (default: 0.1).')

    parser.add_argument('--regular-trade', type=float, default=1e-4,
        help='The trade off of regular_loss (default: 0.01).')               
    parser.add_argument('--regular-type', type=str, default='MLP', choices=['MLP',
        'Flatten_FTF'], help='Type of regularnetwork.')
    
    parser.add_argument('--pretrain', action='store_true',
        help='If backobone network is pretrained.')
    parser.add_argument('--backbone-path', type=str, default=None,
        help='Path to the pretrained backbone.')
    parser.add_argument('--augment', action='store_true', default=False,
        help='Augment the training dataset (default: True).')
    parser.add_argument('--multi-gpu', action='store_true',              
        help='True if use multiple GPUs. Else, use single GPU.')
    parser.add_argument('--num-workers', type=int, default=4,
        help='Number of workers for data loading (default: 1).')
    parser.add_argument('--download', action='store_true',
        help='Download the Omniglot dataset in the data folder.')
    parser.add_argument('--use-cuda', type=bool, default=True,
        help='Use CUDA if available.')
   
    parser.add_argument('--resume', action='store_true', 
        help='Continue from baseline trained model with largest epoch.')
    parser.add_argument('--resume-folder', type=str, default=None,
        help='Path to the folder the resume is saved to.')
    
    args = parser.parse_args()
    
    datasets = ['miniimagenet', 'tieredimagenet', 'cub', 'cifar_fs']
    args.train_data = datasets
    args.train_data.remove(args.test_data)
    
    # make folder and tensorboard writer to save model and results
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    args.record_folder = './{}_{}_{}_{}way_{}shot_{}'.format(args.model_name, args.test_data, args.backbone, str(args.num_ways), str(args.num_shots), cur_time)
    os.makedirs(args.record_folder, exist_ok=True)

    if args.use_cuda and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    elif args.use_cuda:
        raise RuntimeError('You are using GPU mode, but GPUs are not available!')

    # construct model and optimizer
    model = PrototypicalNetwork(args.backbone)
    args.out_channels = get_outputs_c(args.backbone)
    if args.out_channels == 64:
        hh = 10
    elif args.out_channels == 512 or args.out_channels == 640:
        hh = 100
    else:
        hh = 512
    regularnetwork = RegularNetwork(args.out_channels, hh, regular_type=args.regular_type)
    
    if args.use_cuda:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        num_gpus = torch.cuda.device_count()
        if args.multi_gpu:
            model = nn.DataParallel(model)
        model = model.cuda()
        regularnetwork = regularnetwork.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    re_optimizer = torch.optim.Adam(regularnetwork.parameters(), lr=args.omega, weight_decay=0.0005)
    
    # training from the checkpoint
    if args.resume and args.resume_folder is not None:
        # load checkpoint
        checkpoint_path = os.path.join(args.resume_folder, ('_'.join([args.model_name, args.test_data, args.backbone, 'max_acc']) + '_checkpoint.pt.tar'))    
        state = torch.load(checkpoint_path)
        if args.multi_gpu:
            model.module.load_state_dict(state['model'])
        else:
            model.load_state_dict(state['model'])
        train_log = state['train_log']
        optimizer.load_state_dict(state['optimizer'])
        initial_lr = optimizer.param_groups[0]['lr']
        global_task_count = state['global_task_count']
        print('global_task_count: {}, initial_lr: {}'.format(str(global_task_count), str(initial_lr)))
    # training from scratch
    else:
        train_log = {}
        train_log['args'] = vars(args)
        train_log['train_loss'] = []
        train_log['train_acc'] = []
        train_log['val_loss'] = []
        train_log['val_acc'] = []
        train_log['max_acc'] = 0.0
        train_log['max_acc_i_task'] = 0
        initial_lr = args.lr
        global_task_count = 0
        if args.pretrain and args.backbone_path is not None:
            backbone_state = torch.load(args.backbone_path)
            if args.multi_gpu:
                model.module.encoder.load_state_dict(backbone_state)
            else:
                model.encoder.load_state_dict(backbone_state)

    # save the args into .json file
    with open(os.path.join(args.record_folder, 'args.json'), 'w') as f:
        json.dump(vars(args), f)

    # get datasets and dataloaders    
    train_dataloaders = {}
    for i in range(len(args.train_data)):
        train_dataset = get_dataset(args, dataset_name=args.train_data[i], phase='train') 
        train_dataloader = BatchMetaDataLoader(train_dataset, batch_size=args.batch_tasks,
            shuffle=True, num_workers=args.num_workers, pin_memory=True)
        train_dataloaders[args.train_data[i]] = train_dataloader
    val_dataset = get_dataset(args, dataset_name=args.test_data, phase='val')  
    val_dataloader = BatchMetaDataLoader(val_dataset, batch_size=args.batch_tasks,
        shuffle=True, num_workers=args.num_workers)

    with tqdm(total=int(args.train_tasks/args.batch_tasks), initial=int(global_task_count / args.batch_tasks)) as pbar:
        for train_batch_i in range(int(args.train_tasks/args.batch_tasks)):       
            pbar.update(1)
            random_set = random.sample(args.train_data, k=2)
            sd_set = "".join(random_set[0])
            ud_set = "".join(random_set[1:])
            sd_dataloader = train_dataloaders[sd_set]
            ud_dataloader = train_dataloaders[ud_set]
            sd_batch = next(iter(sd_dataloader))
            ud_batch = next(iter(ud_dataloader))

            # chech if lr should decrease as in schedule
            if (train_batch_i * args.batch_tasks) in args.schedule:
                initial_lr *=args.gamma
                for param_group in optimizer.param_groups:
                    param_group['lr'] = initial_lr
            
            global_task_count +=args.batch_tasks

            # dataset donwnload
            sd_support_inputs, sd_support_targets = [_.cuda(non_blocking=True) for _ in sd_batch['train']] if args.use_cuda else [_ for _ in sd_batch['train']]
            sd_query_inputs, sd_query_targets = [_.cuda(non_blocking=True) for _ in sd_batch['test']] if args.use_cuda else [_ for _ in sd_batch['test']]
         
            sd_support_embeddings = model(sd_support_inputs)
            sd_query_embeddings = model(sd_query_inputs)

            sd_prototypes = get_prototypes(sd_support_embeddings, sd_support_targets, train_dataset.num_classes_per_task)
            train_loss = prototypical_loss(sd_prototypes, sd_query_embeddings, sd_query_targets)
            
            regular = regularnetwork(sd_query_embeddings)
            regular_loss = args.regular_trade * regular
            
            optimizer.zero_grad()
            train_loss.backward(retain_graph=True)
            regular_loss.backward(create_graph=True)
            
            if args.multi_gpu:
                grad_theta = [theta_i.grad for theta_i in model.module.parameters()]
                theta_updated_new = {}
                num_grad = 0
                for i, (k, v) in enumerate(model.module.state_dict().items()):
                    if 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k:
                        theta_updated_new[k] = v
                        continue
                    elif grad_theta[num_grad] is None:
                        num_grad +=1
                        theta_updated_new[k] = v
                    else:
                        theta_updated_new[k] = v - initial_lr * grad_theta[num_grad]
                        num_grad += 1
            else:
                grad_theta = [theta_i.grad for theta_i in model.parameters()]
                theta_updated_new = {}
                num_grad = 0
                for i, (k, v) in enumerate(model.state_dict().items()):
                    if 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k:
                        theta_updated_new[k] = v
                        continue
                    elif grad_theta[num_grad] is None:
                        num_grad +=1
                        theta_updated_new[k] = v
                    else:
                        theta_updated_new[k] = v - initial_lr * grad_theta[num_grad]
                        num_grad += 1
            
            model_new = PrototypicalNetwork(args.backbone)
 
            if args.use_cuda:
                if args.multi_gpu:
                    model_new = nn.DataParallel(model_new)
                model_new = model_new.cuda()

            if args.multi_gpu:
                fix_nn(model_new.module, theta_updated_new)
            else:
                fix_nn(model_new, theta_updated_new)

            model_new.train()
            ud_support_inputs, ud_support_targets = [_.cuda(non_blocking=True) for _ in ud_batch['train']] if args.use_cuda else [_ for _ in ud_batch['train']]
            ud_query_inputs, ud_query_targets = [_.cuda(non_blocking=True) for _ in ud_batch['test']] if args.use_cuda else [_ for _ in ud_batch['test']]
            
            ud_support_embeddings = model(ud_support_inputs)
            ud_query_embeddings = model(ud_query_inputs)

            ud_prototypes = get_prototypes(ud_support_embeddings, ud_support_targets, train_dataset.num_classes_per_task)
            
            reward_loss = prototypical_loss(ud_prototypes, ud_query_embeddings, ud_query_targets)
            
            optimizer.step()
            re_optimizer.zero_grad()
            reward_loss.backward()
            re_optimizer.step()

            with torch.no_grad():
                train_accuracy = get_accuracy(prototypes, query_embeddings, query_targets)
                pbar.set_postfix(train_accuracy='{0:.4f}'.format(train_accuracy.item()))
                
            # Validation
            if global_task_count % args.validation_tasks == 0:   
                val_loss_avg = Averager()
                val_acc_avg = Mean_confidence_interval()
    
                model.eval()
                with tqdm(val_dataloader, total=int(args.val_tasks/args.batch_tasks)) as pbar:
                    for val_batch_i, val_batch in enumerate(pbar, 1):

                        if val_batch_i > (args.val_tasks / args.batch_tasks):
                            break

                        support_inputs, support_targets = [_.cuda(non_blocking=True) for _ in val_batch['train']] if args.use_cuda else [_ for _ in val_batch['train']]
                        query_inputs, query_targets = [_.cuda(non_blocking=True) for _ in val_batch['test']] if args.use_cuda else [_ for _ in val_batch['test']]

                        # val loop
                        support_embeddings = model(support_inputs)
                        query_embeddings = model(query_inputs)

                        prototypes = get_prototypes(support_embeddings, support_targets,
                            val_dataset.num_classes_per_task)
                        val_loss = prototypical_loss(prototypes, query_embeddings, query_targets)

                        val_accuracy = get_accuracy(prototypes, query_embeddings, query_targets)
                        pbar.set_postfix(val_accuracy='{0:.4f}'.format(val_accuracy.item()))   
                        val_acc_avg.add(val_accuracy.item()) 
                        val_loss_avg.add(val_loss.item())
                
                val_acc_mean = val_acc_avg.item()
                print('global_task_count: {}, val_acc_mean: {}'.format(str(global_task_count), str(val_acc_mean)))
                if val_acc_mean > train_log['max_acc']:
                    train_log['max_acc'] = val_acc_mean
                    train_log['max_acc_i_task'] = global_task_count
                    save_model(model, args, tag='max_acc')

                train_log['train_loss'].append(train_loss.item())
                train_log['train_acc'].append(train_accuracy.item())
                train_log['val_loss'].append(val_loss_avg.item())
                train_log['val_acc'].append(val_acc_mean)
                save_checkpoint(args, model, train_log, optimizer, global_task_count, tag='max_acc')
    
    # Testing 
    test_dataset = get_dataset(args, dataset_name=args.test_data, phase='test')
    test_dataloader = BatchMetaDataLoader(test_dataset, batch_size=args.batch_tasks,
        shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # construct model and optimizer
    model_path = os.path.join(args.record_folder, ('_'.join([args.model_name, args.test_data, args.backbone, 'max_acc']) + '.pt'))
    state = torch.load(model_path)
    if args.multi_gpu:
        model = nn.DataParallel(model)
        model = model.cuda()
        model.module.load_state_dict(state, False) 
    else:
        model = model.cuda()
        model.load_state_dict(state) 

    test_loss_avg = Averager()
    test_acc_avg = Mean_confidence_interval()
    
    model.eval()
    with tqdm(test_dataloader, total=int(args.test_tasks/args.batch_tasks)) as pbar:
        for test_batch_i, test_batch in enumerate(pbar):
            
            if test_batch_i >= args.test_tasks/args.batch_tasks:
                break

            support_inputs, support_targets = [_.cuda(non_blocking=True) for _ in test_batch['train']] if args.use_cuda else [_ for _ in test_batch['train']]
            query_inputs, query_targets = [_.cuda(non_blocking=True) for _ in test_batch['test']] if args.use_cuda else [_ for _ in test_batch['test']]

            support_embeddings = model(support_inputs)
            query_embeddings = model(query_inputs)

            prototypes = get_prototypes(support_embeddings, support_targets,
                test_dataset.num_classes_per_task)
            test_loss = prototypical_loss(prototypes, query_embeddings, query_targets)

            test_accuracy = get_accuracy(prototypes, query_embeddings, query_targets)
            pbar.set_postfix(test_acc='{0:.4f}'.format(test_accuracy.item()))
            test_acc_avg.add(test_accuracy.item()) 
            test_loss_avg.add(test_loss.item())  
    
    # record
    index_values = [
        'test_acc',
        'best_i_task',    
        'best_train_acc',    
        'best_train_loss',    
        'best_val_acc',
        'best_val_loss'
    ]
    best_index = int(train_log['max_acc_i_task'] / args.validation_tasks) - 1
    test_record = {}
    test_record_data = [
        test_acc_avg.item(return_str=True),
        str(train_log['max_acc_i_task']),
        str(train_log['train_acc'][best_index]),
        str(train_log['train_loss'][best_index]),
        str(train_log['max_acc']),
        str(train_log['val_loss'][best_index]),
    ]
    test_record[args.record_folder] = test_record_data
    test_record_file = os.path.join(args.record_folder, 'record_{}_{}shot.csv'.format(args.test_data, str(args.num_ways), str(args.num_shots)))
    DataFrame(test_record, index=index_values).to_csv(test_record_file)

