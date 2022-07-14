import pickle 
import torch
#network_path = 'networks/afhqdog.pkl'
#network_path = 'networks/afhqcat.pkl'
#network_path = 'networks/metfaces.pkl'
network_path = './ffhq.pkl'
device = 'cuda:0'
with open(network_path, 'rb') as f:
    networks = pickle.Unpickler(f).load()
    
G = networks['G_ema'].to(device)

sample_z = torch.randn(4, 512, device=device)
w = G.mapping(sample_z, None, truncation_psi=1.0)
images = G.synthesis(w)
print('hi')