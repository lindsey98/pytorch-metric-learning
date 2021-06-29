## Project Structure
Copy doc from https://kevinmusgrave.github.io/pytorch-metric-learning/

```
Your Data --> Sampler --> Miner --> Loss --> Reducer --> Final loss value

|_ src/pytorch_metric_learning
  
  |_ samplers: "Samplers are just extensions of the torch.utils.data.Sampler class, i.e. they are passed to a PyTorch Dataloader. The purpose of samplers is to determine how batches should be formed. This is also where any offline pair or triplet miners should exist."
  
  |_ miners: two types of miners
     "Subset Batch Miners take a batch of N embeddings and return a subset n 
      Tuple Miners take a batch of n embeddings and return k pairs/triplets to be used for calculating the loss"
      
  |_ distances
    |_ Cosine similarity
    |_ L-p distance
    |_ Signal to noise ratio (see paper https://arxiv.org/pdf/1904.02616.pdf, 
                              snr = var(hi)/var(hi-hj), hi is anchor feature, hj is compared feature,
                              and d(i,j) = 1/snr )
                              
  |_ losses In particular I am interested in ...
    |_ soft_triple_loss: SoftTripleLoss 
    |_ proxy_losses: ProxyNCALoss
    |_ proxy_anchor_loss: ProxyAnchorLoss  
    
  |_ reducers: "Reducers specify how to go from many loss values to a single loss value"

  |_ trainers: 
  
  |_ regularizers: weight regularizer
  
  |_ testers:
  
  |_ utils:
   |_ accuracy_calculator: supported metric AMI, NMI, mean_average_precision, mean_average_precision_at_r, precision_at_1, precision_at_r (is the 1st neighbor correct? ), r_precision
   |_ inference: "find matching pairs within a batch, or from a set of pairs"
   |_ logging_presets: "ready-to-use hooks for logging data, validating and saving your models, and early stoppage during training"
   |_ common_functions: 
   |_ distributed: "Wrap a loss or miner with these when using PyTorch's DistributedDataParallel (i.e. multiprocessing)"
```
