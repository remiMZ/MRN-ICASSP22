import os
import time
import json
import torch
import random
from tqdm import tqdm
from pandas import DataFrame
import torch.nn.functional as F
from utils import get_accuracy, fix_nn
from torchmeta.modules import DataParallel
from torchmeta.utils.data import BatchMetaDataLoader
from model import MAMLNetwork, RegularNetwork
from torchmeta.utils.gradient_based import gradient_update_parameters
from global_utils import get_dataset, Averager, Mean_confidence_interval, get_outputs_c

def save_model(model, args, tag):
    model_path = os.path.join(args.record_folder, ('_'.join([args.model_name, args.test_data, args.backbone, tag]) + '.pt'))
    if args.multi_gpu:
        model = model.module
    with open(model_path, 'wb') as f:
        torch.save(model.state_dict(), f)

def save_checkpoint(args, model, regularnetwork, train_log, optimizer, global_task_count, tag):
    if args.multi_gpu:
        model = model.module
    state = {
        'args': args,
        'model': model.state_dict(),
        'regularnetwork': regularnetwork.state_dict(),
        'train_log': train_log,
        'val_acc': train_log['max_acc'],
        'optimizer':optimizer,
        'global_task_count': global_task_count
    }
    checkpoint_path = os.path.join(args.record_folder, ('_'.join([args.model_name, args.test_data, args.backbone, tag]) + 'chechpoint.pt.tar'))
    with open(checkpoint_path, 'wb') as f:
        torch.save(state, f)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Domain Generalization Meta-Learning (MRN)')
    parser.add_argument('--model-name', type=str, default='MAML_MRN', help='Name of the model.')
   
    parser.add_argument('--data-folder', type=str, default='./dataset',
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--train-data', type=str, default= 'cub', help='Name of train-data.')
    parser.add_argument('--test-data', type=str, default= 'cub', help='Name of test-data.')
    parser.add_argument('--num-shots', type=int, default= 1, 
        help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-ways', type=int, default= 5, 
        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--test-shots', type=int, default= 15,
        help='Number of examples per class (k in "k-shot", default: 16).')
    parser.add_argument('--backbone', type=str, default='metaconv4',
        help='The type of model backbone.')
    
    parser.add_argument('--batch-tasks', type=int, default= 4,
        help='Number of tasks in a mini-batch of tasks (default: 4).')
    parser.add_argument('--train-tasks', type=int, default=60000,
        help='Number of tasks the model network is trained over (default: 60000).')
    parser.add_argument('--val-tasks', type=int, default=600,
        help='Number of tasks the model network is validated over (default: 600). ')
    parser.add_argument('--test-tasks', type=int, default=10000,
        help='Number of tasks the model network is tested over (default: 10000). The final results will be the average of these batches.')
    parser.add_argument('--validation-tasks', type=int, default=1000,
        help='Number of tasks for each validation (default: 1000).')
    
    parser.add_argument('--lr', type=float, default=0.001,
        help='Initial learning rate (default: 0.001).')
    parser.add_argument('--omega', type=float, default=1e-4,
        help='Initial learning rate of regularnetwork (default: 1e-4).')
    parser.add_argument('--schedule', type=int, nargs='+', default=[15000, 30000, 45000, 60000], 
        help='Decrease learning rate at these number of tasks.')
    parser.add_argument('--gamma', type=float, default=0.1,
        help='Learning rate decreasing ratio (default: 0.1).')

    parser.add_argument('--augment', action='store_true', 
        help='Augment the training dataset (default: True).')
    parser.add_argument('--pretrain', action='store_true',
        help='If backobone network is pretrained.')
    parser.add_argument('--backbone-path', type=str, default=None,
        help='Path to the pretrained backbone.')

    parser.add_argument('--first-order', action='store_true',
        help='Use the first-order approximation of MAML.')
    parser.add_argument('--step-size', type=float, default= 0.4,
        help='Step-size for the gradient step for adaptation (default: 0.4).')
    parser.add_argument('--regular-trade', type=float, default=1e-4,
        help='The trade off of regular_loss (default: 0.01).')               
    parser.add_argument('--regular-type', type=str, default='MLP', choices=['MLP',
        'Flatten_FTF'], help='Type of regularnetwork.')
        
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
    args.out_channels = get_outputs_c(args.backbone)
    model = MAMLNetwork(args.backbone, args.out_channels, args.num_ways)
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
            model = DataParallel(model)
            
        model = model.cuda()
        regularnetwork = regularnetwork.cuda()
        
    optimizer = torch.optim.Adam(model.parameters() , lr=args.lr, weight_decay=0.0005)
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
        regularnetwork.load_state_dict(state['regularnetwork'])
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
                model.module.encoder.load_state_dict(backbone_state, False)
            else:
                model.encoder.load_state_dict(backbone_state, False)

    # save the args into .json file
    with open(os.path.join(args.record_folder, 'args.json'), 'w') as f:
        json.dump(vars(args), f)

    # get datasets and dataloaders
    train_dataloaders = {}
    for i in range(len(args.train_data)):
        train_dataset = get_dataset(args, dataset_name=args.train_data[i], phase='train') 
        train_dataloader = BatchMetaDataLoader(train_dataset, batch_size=args.batch_tasks,
            shuffle=True, num_workers=args.num_workers)      
        train_dataloaders[args.train_data[i]] = train_dataloader
    val_dataset = get_dataset(args, dataset_name=args.test_data, phase='val')
    val_dataloader = BatchMetaDataLoader(val_dataset, batch_size=args.batch_tasks, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    with tqdm(total=int(args.train_tasks/args.batch_tasks)) as pbar:
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
           
            train_loss = torch.tensor(0., device = sd_support_inputs.device)
            train_accuracy = torch.tensor(0., device = sd_support_inputs.device)
            regular_loss = torch.tensor(0., device = sd_support_inputs.device)
            for _ , (sd_support_input, sd_support_target, sd_query_input, sd_query_target) in enumerate(zip(sd_support_inputs, sd_support_targets, sd_query_inputs, sd_query_targets)):
                sd_support_predictions = model(sd_support_input)
                sd_inner_loss = F.cross_entropy(sd_support_predictions, sd_support_target)

                model.zero_grad()
                params = gradient_update_parameters(model, sd_inner_loss, step_size=args.step_size,
                    first_order=args.first_order)

                sd_query_features, sd_query_predictions_outer = model(sd_query_input, params=params, Emd=True)
                train_loss += F.cross_entropy(sd_query_predictions_outer, sd_query_target)

                regular = regularnetwork(sd_query_features)
                regular_loss += args.regular_trade * regular
                
                with torch.no_grad():
                    train_accuracy += get_accuracy(sd_query_predictions_outer, sd_query_target)

            regular_loss.div_(args.batch_tasks)
            train_loss.div_(args.batch_tasks)
            train_accuracy.div_(args.batch_tasks)
            
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
            
            model_new = MAMLNetwork(args.backbone, args.out_channels, args.num_ways)
 
            if args.use_cuda:
                if args.multi_gpu:
                    model_new = DataParallel(model_new)
                model_new = model_new.cuda()

            if args.multi_gpu:
                fix_nn(model_new.module, theta_updated_new)
            else:
                fix_nn(model_new, theta_updated_new)

            model_new.train()
            ud_support_inputs, ud_support_targets = [_.cuda(non_blocking=True) for _ in ud_batch['train']] if args.use_cuda else [_ for _ in ud_batch['train']]
            ud_query_inputs, ud_query_targets = [_.cuda(non_blocking=True) for _ in ud_batch['test']] if args.use_cuda else [_ for _ in ud_batch['test']]
            
            reward_loss = torch.tensor(0., device = sd_support_inputs.device)
            for _ , (ud_support_input, ud_support_target, ud_query_input, ud_query_target) in enumerate(zip(ud_support_inputs, ud_support_targets, ud_query_inputs, ud_query_targets)):
                ud_support_predictions = model_new(ud_support_input)
                ud_inner_loss = F.cross_entropy(ud_support_predictions, ud_support_target)
                
                model_new.zero_grad() 
                params = gradient_update_parameters(model_new, ud_inner_loss, step_size=args.step_size,
                    first_order=args.first_order)

                ud_query_predictions = model_new(ud_query_input, params=params)
                reward_loss += F.cross_entropy(ud_query_predictions, ud_query_target)

            reward_loss.div_(args.batch_tasks)

            optimizer.step()
            re_optimizer.zero_grad()
            reward_loss.backward()
            re_optimizer.step()
            
            with torch.no_grad():
                train_accuracy = get_accuracy(sd_query_predictions_outer, sd_query_targets)
                pbar.set_postfix(train_accuracy='{0:.4f}'.format(train_accuracy.item()))
                
            # Validation
            if global_task_count % args.validation_tasks == 0:   
                val_loss_avg = Averager()
                val_acc_avg = Mean_confidence_interval()
  
                with tqdm(val_dataloader, total=int(args.val_tasks/args.batch_tasks)) as pbar:
                    for val_batch_i, val_batch in enumerate(pbar, 1):
                        if val_batch_i > (args.val_tasks / args.batch_tasks):
                            break
                        support_inputs, support_targets = [_.cuda(non_blocking=True) for _ in val_batch['train']] if args.use_cuda else [_ for _ in val_batch['train']]
                        query_inputs, query_targets = [_.cuda(non_blocking=True) for _ in val_batch['test']] if args.use_cuda else [_ for _ in val_batch['test']]

                        model.eval()
                        val_loss = torch.tensor(0., device = support_inputs.device)
                        val_accuracy = torch.tensor(0., device = support_inputs.device)
 
                        for _ , (support_input, support_target, query_input, query_target) in enumerate(zip(support_inputs, support_targets, query_inputs, query_targets)):
                            support_predictions = model(support_input)
                            inner_loss = F.cross_entropy(support_predictions, support_target)
                                
                            model.zero_grad()
                            params = gradient_update_parameters(model, inner_loss, step_size=args.step_size,
                                    first_order=args.first_order)
 
                            with torch.no_grad():
                                query_predictions = model(query_input, params=params)
                                val_loss += F.cross_entropy(query_predictions, query_target)
                                val_accuracy += get_accuracy(query_predictions, query_target)

                        val_loss.div_(args.batch_tasks)
                        val_accuracy.div_(args.batch_tasks)
                            
                        pbar.set_postfix(val_acc='{0:.4f}'.format(val_accuracy.item()))
                        val_loss_avg.add(val_loss.item())
                        val_acc_avg.add(val_accuracy.item())

                # record
                val_acc_mean = val_acc_avg.item()
                print('global_task_count: {}, val_acc_mean: {}'.format(str(global_task_count), str(val_acc_mean)))
                if val_acc_mean > train_log['max_acc']:
                    train_log['max_acc'] = val_acc_mean
                    train_log['max_acc_i_task'] = global_task_count
                    save_model(model, args, tag='max_acc')

                train_loss_main = train_loss + regular_loss
                train_log['train_loss'].append(train_loss_main.item())
                train_log['train_acc'].append(train_accuracy.item())
                train_log['val_loss'].append(val_loss_avg.item())
                train_log['val_acc'].append(val_acc_mean)

                save_checkpoint(args, model, regularnetwork, train_log, optimizer, global_task_count, tag='max_acc')
                
    # testing...
    test_dataset = get_dataset(args, dataset_name=args.test_data, phase='test')
    test_dataloader = BatchMetaDataLoader(test_dataset, batch_size=args.batch_tasks, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # construct model and optimizer
    model_path = os.path.join(args.record_folder, ('_'.join([args.model_name, args.test_data, args.backbone, 'max_acc']) + '.pt'))
    if args.multi_gpu:
        model = DataParallel(model)
        model = model.cuda()
        model.module.load_state_dict(state, False) 
    else:
        model = model.cuda()
        model.load_state_dict(state) 
    
    test_loss_avg = Averager()
    test_acc_avg = Mean_confidence_interval()
    
    with tqdm(test_dataloader, total=int(args.test_tasks/args.batch_tasks)) as pbar:
        for test_batch_i, test_batch in enumerate(pbar):
            if test_batch_i >= args.test_tasks/args.batch_tasks:
                break

            support_inputs, support_targets = [_.cuda(non_blocking=True) for _ in test_batch['train']] if args.use_cuda else [_ for _ in test_batch['train']]
            query_inputs, query_targets = [_.cuda(non_blocking=True) for _ in test_batch['test']] if args.use_cuda else [_ for _ in test_batch['test']]

            model.eval()
            test_loss = torch.tensor(0., device = support_inputs.device)
            test_accuracy = torch.tensor(0., device = support_inputs.device)
            for _ , (support_input, support_target, query_input, query_target) in enumerate(zip(support_inputs, 
                support_targets, query_inputs, query_targets)):
                support_predictions = model(support_input)
                inner_loss = F.cross_entropy(support_predictions, support_target)
                        
                model.zero_grad()
                params = gradient_update_parameters(model, inner_loss, step_size=args.step_size,
                        first_order=args.first_order)

                with torch.no_grad():
                    query_predictions = model(query_input, params=params)
                    test_loss += F.cross_entropy(query_predictions, query_target)
                    test_accuracy += get_accuracy(query_predictions, query_target)

            test_loss.div_(args.batch_tasks)
            test_accuracy.div_(args.batch_tasks)
            pbar.set_postfix(test_acc='{0:.4f}'.format(test_accuracy.item()))
            test_loss_avg.add(test_loss.item())
            test_acc_avg.add(test_accuracy.item())
        
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
    test_record_file = os.path.join(args.record_folder, 'record_{}_{}way_{}shot.csv'.format(args.test_data, str(args.num_ways), str(args.num_shots)))
    DataFrame(test_record, index=index_values).to_csv(test_record_file)

    
    
    
    
    
    
    
    
    
    
    
    
  
    

   
    
