import argparse
import torch
import os
from models.Inception import InceptionV3
from utils.attack_gan_fid_latents import attack_gan_fid
from utils import all_datasets
from models.resnet import resnet
from models.robust_inception import robust_inceptionv3
from models.gan_wrapper import StyleGAN, GAN_Wrapper
device = 'cuda' if torch.cuda.is_available else 'cpu'

def main(args):
    
    #Loading Generator.
    generator = StyleGAN(args.dataset, args.truncation)

    #Loading Discriminator.
    if args.robust_inception_path is None:
        descriminator = InceptionV3() if args.model == 'inception' else resnet()
    else:
        print("Robust Inception is being used")
        descriminator = robust_inceptionv3(args.robust_inception_path)
    
    # Putting the generateor and discriminator in one wrapper
    model = GAN_Wrapper(generator, descriminator)
    model.eval().requires_grad_(False)

    # Initializing the attacker
    attacker = attack_gan_fid(model, lr=args.lr, steps=args.iters,
                             batch_size=args.batch_sz, device=device)
    
    # Loadig the embedding of the real dataset. Computing the embedding should be done in a pre-processing step.
    real_dataset = torch.load(args.real_latents + '/embeddings.pth').to("cpu")
    
    if args.evaluate_path is not None: #Evaluating an existing dataset
        fake_dataset = torch.load(args.evaluate_path + '/embeddings.pth')
    else:  #Embeddings based on FID-guided sampling.
        fake_dataset = attacker.run_attack(real_dataset, args.eps, args.output_path, args.optimize_w)
    
    # Evaluating the FID
    f = open(args.output_path + '/fid.txt', 'w')
    print("fid\tnum-instances", file=f, flush=True)
    fid_components = attacker.compute_fid(fake_dataset, real_dataset, return_parts=True)
    fid = fid_components[0] + fid_components[1]
    print('{}={}+{}\t{}'.format(fid, fid_components[0], fid_components[1],
        len(fake_dataset)), file=f, flush=True)
    print('Result: {} = {} + {}'.format(fid.item(), fid_components[0].item(), fid_components[1].item()))
    print(fid, fid_components)
    return

if __name__=='__main__':
    dataset_choices = ['metfaces', 'dog', 'cat', 'ffhq']
    split_choices = ['train', 'val']
    model_choices = ['resnet', 'inception']
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output-path', type=str, default='./', required=True,
                     help='path to save the fake dataset')
    parser.add_argument('--model', type=str, default='inception', choices=model_choices,
                    help='model used to compute the evaluating metric')
    parser.add_argument('--dataset', type=str, default='ffhq', choices=dataset_choices,
                    help='dataset of that the GAN was trained on')

    # Real dataset path  
    parser.add_argument('--real-latents', type=str, default='./', required=True,
                    help='path to the real world dataset to compute distance with respect to')
    
    # Pretrained path for Inception
    parser.add_argument('--robust-inception-path', type=str, default=None,
                        help='path to the robust inception path')

    # Arguments for the StyleGAN
    parser.add_argument('--truncation', type=float, default=1.0,
                    help='batch size for to run the optimization')

    # Optimization argument
    parser.add_argument('--batch-sz', type=int, default=64,
                    help='batch size for to run the optimization')
    parser.add_argument('--iters', type=int, default=100,
                    help='number of iterations for the optimization')                
    parser.add_argument('--lr', type=float, default=0.5,
                    help='learning rate for the optimization')
    parser.add_argument('--eps', type=float, default=None,
                    help='radius of lp ball for adversry. Usually it is set to 1.0')

    parser.add_argument('--optimize-w', default=False, action='store_true',
                    help='run the minimization in the latent space')

    # Evaluating arguments without optimization
    parser.add_argument('--evaluate-path', type=str, default=None,
                    help='path to dataset to compute inception score for.\
                         If None then it will create a fake dataset')
        
    args = parser.parse_args()

    if not os.path.exists(args.output_path): #To save the results
        os.makedirs(args.output_path)

    main(args)
