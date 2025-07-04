import torch
import torch.nn as nn


class SimCLRv2(nn.Module):
    """
    Paper: https://arxiv.org/pdf/2006.10029.pdf
    Adapt from https://github.com/nikheelpandey/TAUP-PyTorch/blob/master/loss.py

    """
    def __init__(self, temp=0.5, normalize= False):
        super(SimCLRv2, self).__init__()
        self.temp = temp
        self.normalize = normalize
    
    def forward(self,xi,xj):

        z1 = F.normalize(xi, dim=1)
        z2 = F.normalize(xj, dim=1)
        
        N, Z = z1.shape 
        device = z1.device 
        
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = torch.mm(representations, representations.T)

        # create positive matches
        l_pos = torch.diag(similarity_matrix, N)
        r_pos = torch.diag(similarity_matrix, -N)
        positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)


        # get the values of every pair that's a mismatch
        diag = torch.eye(2*N, dtype=torch.bool, device=device)
        diag[N:,:N] = diag[:N,N:] = diag[:N,:N]        
        negatives = similarity_matrix[~diag].view(2*N, -1)

        
        logits = torch.cat([positives, negatives], dim=1)
        logits /= self.temp
        labels = torch.zeros(2*N, device=device, dtype=torch.int64)
        loss = F.cross_entropy(logits, labels, reduction='sum')
        
        return (loss / (2 * N))
