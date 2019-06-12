**Variational Auto Encoders**

(Vanilla) Autoencoders have the intermediate latent space which interface Encoder and Decoder. This latent space is a $\mathbb{R}$eal valued space. The Encoder can map input $X$ to any arbitrary real valued $F$ from which the decoder can reconstruct original input. 

Next, we look inside this latent space $F$ and make some assumptions(for fun, better understanding, or just simpler interfacing of encoder and decoder). Maybe the latent space need not to be any arbitrary real valued space. Maybe its a random variable with its distributional parameters that can be either guessed, or learned. What we have now with this assumption is Variational AutoEncoder(VAE). VAEs have a strong assumption on the latent space.

Lets start off with a guess: mabe the latent space representation is a multivariate Gaussian(Mean, Variance), normalized to Zero mean and Unit variance i.e. $N(0, I)$. There has been some work to relax this constraint by learning these parameters of latent space distribution as well. (NOTE: Not detouring to those line of work here.) 

Since we made a strong assumption on the latent space, we need to enforce it on encoders and decoders. Here is an idea: We just need to enforce the encoder to obey this assumption, and there is no change needed to the decoder since it will work with the constrained encoder just like before. How to do this? We modify the loss function slightly to enforce the assumption.

Here is a walk through:

0. Assumption: There is a distribution $z$ from which all samples are generated, i.e $p(z)$ are its parameters. It could generate all samples in the universe, and those could be infinite! We have a small subset of those samples at hand $x \in X$ which we call our training set.
0. However, we dont know the parameters of full $p(z)$, so we are trying to estimate it. 
  + $p(z)$ which can generate almost anything in the universe is hard to estimate. 
  + Let's take a constrained version of this: we have a finite set of samples $x \in X$ (our dataset), so the joint $P(z, x)$ sounds easier to approximate and validate. $p(z)$ can be seen as marginalization of $p(z, x)$ over an infinite set of samples $x$ in the universe (that sounds hard). 
0. We know $p(z, x) = p(x) \times p(z \vert x) = p(z) \times p(x \vert z)$.
  + Interesting Observation: suppose we have a way to force $p(z \vert x) \approx p(z)$ (we do, as seen later), then $p(x \vert z) \approx p(x)$
0. Encoder $Enc_\phi : x \rightarrow z$ is same as $Enc_\phi(z \vert x)$ an approximation to $p(z \vert x)$, where $p(z \vert x)$ itself an approximation to $p(z)$
0. Decoder $Dec_\psi : z \rightarrow x$ is same as $Dec_\psi(x \vert z)$ the reconstruction from latent space. this is an approximation to $p(x \vert z)$ which itself an approximation to $p(x)$. this approximation holds iff $Enc_\phi(z \vert x) \approx p(x \vert z) \approx p(z)$
0. $ReconstructionLoss$ is same as vainilla autoencoder, and it depends on the task at hand:
 + squared error: $\Vert X − Dec_\psi(Enc_\phi(X))\Vert^2$
 + cross entropy (in some variation): $-log Dec_\psi(Enc_\phi(X))$ (for time series classification like text sequence)
0. We want the $Enc_\phi(z \vert x)$ to be close to $p(z)$ (or $p(z \vert x)$). We dont have $p(z)$  or $p(z \vert x)$. Maybe we can learn an approximation to it: $p(z) \approx p_\theta(z)$. We can learn $p_\theta(z)$ later, for now (the simple model) take a safe guess $p_\theta(z) = N(0, I)$
0. $TotalLoss = D_{KL} ( Enc_\phi(z \vert x) \Vert p_\theta(z)) + ReconstructionLoss$ 


$D_{KL}$ is the KL divergence between what encoder produces $Enc_\phi(z \vert x)$ and 
the assumed+learned latent space $p_\theta(z)$. Diveregence measure enfoces the distributions to look similar, i.e. $Enc_\phi(z \vert x) \approx p_\theta(z)$. We can replace $D_{KL}$ with something else as long as it forces the distributions to look same. It can be any other divergence measure if that helps in the robust convergence of loss during training. 
> For example, $Wasserstein Distance-1$ can be used if constraining the parameters in $Enc_\phi$ to a certain range. Recently $Wasserstein Distance-1$ with some constraints on $Enc_\phi$ has become popular. 
> TODO: read more Wasserstein AutoEncoder.
