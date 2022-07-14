import os
import torch
import argparse
from utils import all_datasets
from models.resnet import resnet
from models.Inception import InceptionV3
from utils.attacks import attack_inception_score
from models.robust_inception import robust_inceptionv3

device = 'cuda' if torch.cuda.is_available else 'cpu'

def main(args):
    #Loading the model
    if args.robust_inception_path is None:
        model = InceptionV3() if args.model == 'incpetion' else resnet()
    else:
        model = robust_inceptionv3(args.robust_inception_path)
    
    model.eval().requires_grad_(False)

    #Loading the attacker
    attacker = attack_inception_score(model, lr=args.lr, steps=args.iters,
        threshold=args.threshold, batch_size=args.batch_sz, device=device)
    
    #Evaluating a given dataset
    if args.evaluate_path is not None: #Evaluating an existing dataset
        new_dataset = all_datasets.Fake_Dataset(args.evaluate_path)
        print("Your dataset is loaded. Now the inception score will be computed")
    else:  #Calling predefined datasets (CIFAR10 and ImageNet)
        if hasattr(all_datasets, args.dataset):
            get_dataset = getattr(all_datasets, args.dataset)
            dataset = get_dataset(args.dataset_path, args.dataset_split)

        else: # Create random noise dataset
            dataset = all_datasets.random_dataset(args.num_instances, args.num_classes, args.resolution)       
        
        #Modifying the dataset by running an attack
        new_dataset = attacker.run_attack(dataset, args.eps, args.output_path)
    
    # Evaluating the Inception score of the dataset
    f = open(args.output_path + '/inception_score.txt', 'w')
    print("IS_mean\tIS_std\tsplits\tnum-instances", file=f, flush=True)
    inception_score, std = attacker.compute_inception_score(new_dataset, splits=args.splits)
    print('{}\t{}\t{}\t{}'.format(inception_score, std, args.splits,
                                     len(new_dataset)), file=f, flush=True)
    
    return

if __name__=='__main__':
    dataset_choices = ['cifar10', 'imagenet']
    split_choices = ['train', 'val']
    model_choices = ['resnet', 'inception']
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output-path', type=str, default='./',
                        help='path to save the fake dataset')
    parser.add_argument('--model', type=str, default='inception', choices=model_choices,
                    help='model used to compute the evaluating metric')

    # Pretrained path for Inception
    parser.add_argument('--robust-inception-path', type=str, default=None,
                        help='path to the robust inception path')
    
    # Fake Dataset arguments
    parser.add_argument('--num-instances', type=int, default=10000,
                    help='number of instances in the adversary dataset')
    parser.add_argument('--num-classes', type=int, default=1008,
                    help='number of classes in the adversary dataset')
    parser.add_argument('--resolution', type=int, default=299,
                    help='resulotion of the image in the adversary dataset')

    # Known Dataset arguments
    parser.add_argument('--dataset', type=str, default='', choices=dataset_choices,
                    help='Dataset to initialize with. if None then random data will be used.')
    parser.add_argument('--dataset-path', type=str, default='./',
                    help='path to dataset to be used')
    parser.add_argument('--dataset-split', type=str, default='val', choices=split_choices,
                    help='Split of the dataset to be used. Either train or val')

    # Optimization argument
    parser.add_argument('--batch-sz', type=int, default=64,
                    help='batch size for to run the optimization')
    parser.add_argument('--iters', type=int, default=100,
                    help='number of iterations for the optimization')                
    parser.add_argument('--lr', type=float, default=0.5,
                    help='learning rate for the optimization')
    parser.add_argument('--threshold', type=float, default=0.99,
                    help='threshold for the optimization')
    parser.add_argument('--eps', type=float, default=None,
                    help='radius of lp ball for adversry (\|\delta\|_\infty \leq eps).')

    # Evaluating Inception score arguments
    parser.add_argument('--splits', type=int, default=1,
                    help='batch size for to run the optimization')
    parser.add_argument('--evaluate-path', type=str, default=None,
                        help='path to dataset to compute inception score for.\
                         If None then it will create a fake dataset')        
    args = parser.parse_args()

    if not os.path.exists(args.output_path): #To save the inception score results
        os.makedirs(args.output_path)

    if args.evaluate_path is None:
        for i in range(args.num_classes):
            if not os.path.exists(args.output_path+ '/fake_images/{}'.format(i)):
                os.makedirs(args.output_path+ '/fake_images/{}'.format(i))
    main(args)
