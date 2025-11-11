import torch.nn as nn
from args import args
import torch
class MSE:
    def __init__(self, model, latent_loss_weight=0.25,dist_penalty_weight=0.25, dist_penalty_topk=1):
        self.model = model
        self.latent_loss_weight = latent_loss_weight
        self.dist_penalty_weight = dist_penalty_weight
        self.dist_penalty_topk = dist_penalty_topk
        self.mse = nn.MSELoss()

    def compute(self, batch,batch_y, details=False,dist_penalty=False):
        seqs = torch.cat([batch,batch_y],dim=1)
        out, latent_loss, _ = self.model(batch,batch_y)
        recon_loss = self.mse(out, seqs)
        latent_loss = latent_loss.mean()
        loss = recon_loss + self.latent_loss_weight * latent_loss
        if dist_penalty:
            codebook_weight = self.model.quantize.embedding.weight
            _detach_weight = codebook_weight.detach()

            dist = torch.cdist(codebook_weight, _detach_weight, p=2)
            diagonal_matrix = torch.eye(dist.shape[0]) * 1e9

            dist = dist + diagonal_matrix.to(dist.device)

            min_dist, _ = torch.topk(dist, self.dist_penalty_topk, dim=-1, largest=False)
            min_dist = 1. / (min_dist + 1e-5)

            dist_penalty_value = min_dist.mean()

            loss = loss + self.dist_penalty_weight * dist_penalty_value
        if details:
            return {
                'loss': loss,
                'recon_loss': recon_loss,
                'latent_loss': latent_loss
            }
        else:
            return loss