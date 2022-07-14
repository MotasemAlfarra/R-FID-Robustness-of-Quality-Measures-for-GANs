import argparse
import torch
import os
from models.Inception import InceptionV3
from utils.attack_fid import attack_fid
from utils import all_datasets
from models.resnet import resnet
from models.robust_inception import robust_inceptionv3

device = 'cuda' if torch.cuda.is_available else 'cpu'

def main(args):
    #Loading the model
    if args.robust_inception_path is None:
        model = InceptionV3() if args.model == 'inception' else resnet()
    else:
        model = robust_inceptionv3(args.robust_inception_path)    
    
    model.eval().requires_grad_(False)

    #Loading the attacker
    attacker = attack_fid(model, lr=args.lr, steps=args.iters,
        threshold=args.threshold, batch_size=args.batch_sz, device=device)
    
    #Loading the real dataset (Cifar10 or ImageNet)
    real_dataset = all_datasets.Fake_Dataset(args.real_dataset_path)

    if args.evaluate_path is not None: #Evaluating an existing dataset
        fake_dataset = all_datasets.Fake_Dataset(args.evaluate_path)
    
    else:  #Loading a dataset
        if args.fake_dataset_path is not None:
            dataset = all_datasets.Fake_Dataset(args.fake_dataset_path)

        else: # Create random noise dataset
            dataset = all_datasets.random_dataset(args.num_instances, args.num_classes, args.resolution)       
        
        fake_dataset = attacker.run_attack(dataset, real_dataset, args.eps, args.output_path)
    
    # Evaluating the Inception score
    f = open(args.output_path + '/fid.txt', 'w')
    print("fid\tnum-instances", file=f, flush=True)
    fid = attacker.compute_fid(fake_dataset, real_dataset)
    print('{}\t{}'.format(fid, len(fake_dataset)), file=f, flush=True)
    print(fid)
    return

if __name__=='__main__':
    dataset_choices = ['cifar10', 'imagenet']
    split_choices = ['train', 'val']
    model_choices = ['resnet', 'inception']
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output-path', type=str, default='./', required=True,
                     help='path to save the fake dataset')
    parser.add_argument('--model', type=str, default='inception', choices=model_choices,
                    help='model used to compute the evaluating metric')
    parser.add_argument('--real-dataset-path', type=str, default='./', required=True,
                    help='path to the real world dataset to compute distance with respect to')
    parser.add_argument('--fake-dataset-path', type=str, default=None, 
                    help='path to the fake dataset to maximize distance with respect to')
    
    # Pretrained path for Inception
    parser.add_argument('--robust-inception-path', type=str, default=None,
                        help='path to the robust inception path')
    
    # Fake Dataset arguments (Random)
    parser.add_argument('--num-instances', type=int, default=10000,
                    help='number of instances in the adversary dataset')
    parser.add_argument('--num-classes', type=int, default=1008,
                    help='number of classes in the adversary dataset')
    parser.add_argument('--resolution', type=int, default=299,
                    help='resulotion of the image in the adversary dataset')

    # Optimization argument
    parser.add_argument('--batch-sz', type=int, default=64,
                    help='batch size for to run the optimization')
    parser.add_argument('--iters', type=int, default=100,
                    help='number of iterations for the optimization')                
    parser.add_argument('--lr', type=float, default=0.5,
                    help='learning rate for the optimization')
    parser.add_argument('--threshold', type=float, default=0.99,
                    help='number of iterations for the optimization')
    parser.add_argument('--eps', type=float, default=None,
                    help='radius of lp ball for adversry. Usually it is set to 8/255')

    # Evaluating arguments without optimization
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