import torch

def compute_mean_mad(dataloader, device):
    ####for SOAP
    
    values = torch.load('/home/snirhordan/GramNN/labels.pt').to(device, dtype=torch.float)
    length = len(values.tolist())
    train_length = torch.ceil(torch.tensor(length) * 0.8)
    values = values[:int(train_length)]
    print(values.size())
    meann = torch.mean(values)
    ma = torch.abs(values - meann)
    mad = torch.mean(ma)
    return meann, mad