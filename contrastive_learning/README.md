1. change the dataset[2] to any other dataset
2. run command: python contrastive_learning.py

We use sergio to simulate noisy data. Then, we use the noisy data D_n as the input. We first normalize the input D_n to make sure it Scale to [-1, 1] for Tanh activation used in VAE. We then do permutation across rows, randomly select a indice list with size of batch_size for each batch, and take those rows as a batch for training. We then train the VAE 1500 epoches to autoencode the gene expressions. The loss decreased .

The loss function in training is consisted of three parts.
1. KL divergence loss. Minimize this to better predict the distribution of latent variable.
2. reconstruction loss (MSE). Minimize this to decrease the discrepancy between the original data and reconstructed representation, so that the model can potentially capture the nuances of the input data.
3. soft-nearest neighbors loss. https://lilianweng.github.io/posts/2021-05-31-contrastive/ This is a contrastive loss that use all positive and negative point in the batch.

Todos:
1. measure the initial condition (H), the minimized version, and the maximized version to see if the loss function works as expected. color the different classes. we should plot it, and I am expecting to see that the connected edges should come together closer.
2. different termperature can significantly impact the performance. eliminate temperature
as a hyperparameter by defining the entanglement loss as the minimum value over all temperature. We can approximate this quantity by initializing T to a predefined value and, at every calculation of the loss, optimizing with gradient descent over T to minimize the loss. In practice, we found optimization to be more stable when we learn the inverse of the temperature. this is from the paper and i don't know how to do this.
3. think about whether we want to maximize or minimize the loss here. the data is gene expression data and we assume that genes that are connected in the GRN
4. we should measure that given a directed edge from gene1 to gene2, how likely is this edge exist. those genes that have edges bewteen each other in the validation and test set should have higher average probability than those which don't have.
5. i am not sure how to include the direciton property in the training process so that the model can know the sequence, just like language we value order of text "i don't" and "don't i" is somehow different.

GENIE3 is a good method.


6. test the impact of strong and weak connection
    ds2 from 0.50 to 0.518
    still need testing, also w
    
7. why there's only one cluster in the result?
8. plot the embedding space and then draw the 
9. relationship between regulatory info from interactions file and grn file.
    yunwei: I computed the aucroc between imputed one and interactions gt and the aucroc between the same imputed one with the compulete GRN. It turns out that the results are same.
10. change the network structure
    yunwei: hidden dim from 1000 to 1024, latent dim from 128 to 256, batch_size from 512 to 1024
    it turns out the performance is really bad. DS1 drop from 0.5 to 0.38, DS2 drop from 0.55 to 0.45
    so what happened?
    On DS1, change it back to 512 gives us: 0.4530, change it to 256 gives us: 0.6115. change it to 128 gives us: 0.5076. where the aucroc on clean data is around 0.7 and noisy data is around 0.5.
    To avoid overfitting, we expect that at least training set is lower than the validation set. To approach this, we drop the nun epoch from 1500 to 500. It is not overfitting anymore but it has a bad performance. it gives us 0.48. Then increase it to 1000, it is overfitted and it gives us 0.52. If increase to 3000, the performance is also very bad, at around 0.47. Redo the experiment on 1500, then it gives us
    This doesn't work well in DS1, since the performance is quite random.

Redifine the structure:
1. Set dataset G_train, G_valid, G_test
     this is achieved by load_ground_truth_grn, sample_partial_grn, split_train_valid, and get_test_set 
2 VAE
    Adapt to different VAE
3. Train VAE model to get the latent representation
    for each batch, give a balanced number of edges. total positive edges / num_batches, we should be able to get the same of edges by randomlly form false non-exist edges. you can get edges from an adajcency matrix which should be built based on directional edgesm which means if there's an edge form gene1 to gene2, gene1 to gene2 would be 1 but the other way would not.
4. Train contrastive learning algo to get a score

the kld will increase from 0 to ~4 and the reconstruction loss maybe decrease from 400 to 50

It is common for KLD to first increase and then decrease during training. Initially, as the encoder learns to represent the data, KLD may rise because the latent distributions start diverging from the prior. As training progresses and the model balances reconstruction accuracy with the latent space regularization (due to the KLD term in the loss function), KLD often decreases.

The reconstruction loss should steadily decrease over time as the model learns to reconstruct the input data more accurately.