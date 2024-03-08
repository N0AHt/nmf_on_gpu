# Code for running non negaive factorization on GPU in parallel (hopefully faster than sklearn)

Runs both a jax and a pytorch env to compare performances


## Baseline:
With sklearn decomposition's implementation of nmf, factorising a sample dataset takes over 25 mins for a model
with rank 100 (100 features)

## In pytorch
using the module torchnmf, running factorisation on a sample dataset (torch.Size([382, 777600])) takes around 6s for the same model
 
