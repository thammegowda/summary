# Generative Adversarial Networks(GANs)

As we seen in the previous section, the Variation Auto Encoder made a strong assumption of latent space:
1. Latent space can be modelled as a random distribution
2. That random distribution is a multivariate gaussian with zero mean and unit variance.

The role of an encoder in the variational autoencoder is to convert any input $x$ into some point in the space of $\mathcal{N}(0, I)$. In other words, if we have a sample set of examples (images, text), each of them can be deterministically mapped (using an encoder) to a point in the space $\mathcal{N}(0, I)$. 
We have one unique point for each example in our finite dataset.
However, there are inifinite number of points in $\mathcal{N}(0, I)$, there fore we could in theory generate infinite number of examples using the $Dec\psi(z)$ where $z$ is a sample from $\mathcal{N}(0, I)$.

Since the desired output space of the encoder is known/assumed to be $\mathcal{N}(0, I)$, and we do know how to generate a random point from that space, then why do we need an encoder? Especially if we are just interested in generation from this known latent space, we can take out the encoder, since all we need is a decoder to convert noise to an example.
In this model-without-Encoder, let us call the `Decoder` as `Generator` since it is more meaningful name (and thats what the authors called it).

To begin with, we obtain some random samples from $\mathcal{N}(0,I)$ (call it $noise$) and pass it to the Decoder aka Generator. The generator would always generate something irrespective of what we provide as input. But the real challenge here is making the generator generate only useful outputs. Since there is no encoder in this model, we dont know which specific example was mapped to the noise, so we are unable to measure reconstruction loss. Most of the time it is not even reconstruction, it is a novel/never-before-seen construction. This model needs a clever way to force the generator produce only useful outputs.

{% cite goodfellow2014GAN %} had a clever and interesting idea which is now commonly referred as *adversarial training*.  

> I see an interesting connection to algorithms complexity theory: Suppose if we have a blackbox to efficiently decide a given solution is correct or wrong (i.e. NP aka Non-deterministic Polynomial), searching a correct solution of a NP (hard) problem can be done efficiently using this deciding blackbox. Here, if we can have a blackbox to decide the generated output as Good or Bad, we can use it to search the good parameters of Generator.

The module that can signal whether a generated output is good or bad is called *Descriminator*. Since there are no readily available descriminator, it needs to be learned too. The Descriminator can also be treated as a parameterized function, and learn its parameters.
Since we are implementing parameterized functions as neural networks, we can connect Decoder aka Generator to Descriminator to form a bigger network and optimize both of them as one end objective. There is more detail involved to the optimization technique, it is more interesting than it looks at surface.


Walk through for how the generator-descriminator network can be adversarially optimized
> the authors called generator as $g_\theta$, to be consistent with previous notes on VAEs, $g_\theta = Gen_\psi = Dec_\psi$. 

0. The assumed latent space $\mathcal{N}(0, I)$
0. Sample a radom point from the latent space, say $s$. There is an example  corresponding to this $s$. The generator can generate it. 
  + the sample space $\mathcal{N}(0, I)$ has infinite samples. Even if we obtain a very large finite set of known examples the probability of $s$ corresponding to a known example in the set is negligible. That is okay, since we are not after exact reconstruction loss
0. Generator: $\tilde{z} = Gen_\psi(s)$ Here $\tilde{z}$ is a generated example
0. Descriminator:  $r=Desc_w(\tilde{z})$ . The range of $r=[0, 1]$ Where 0 mean bad/useless/fake and 1 means good/useful/real.
0. Simultaneously train the $Gen_\psi$ and $Desc_w$. To simultaneously optimize all the parameters, we will need a set of realistic examples.
0. Notation: Let $z$ be the real example read from training set (whose space is $P_*$), and $\tilde{z}= Gen_\psi(s)$  with space $P_{\tilde{z}}$
0. A well trained Descriminator should have $Desc_w(z)$ to be close to 1 (since they are real), and $Desc_w(\tilde{z})$ should be close to $0$ since are generated (or fake)
0. To get to the well trained Descriminator, maximimize $E_{z \sim P_*} \space [ Desc_w(z) ] - E_{\tilde{z} \sim P_z} \space [ Desc_w(\tilde{z}) ]$
0. Meanwhile the Generator's task is to get better at making the descriminator's task harder. Therefore, the objective for Generator is to do exactly opposite of Descriminator (a true adversary), which is to minimize $E_{z \sim P_*} \space [ Desc_w(z) ] - E_{\tilde{z} \sim P_z} \space [ Desc_w(\tilde{z}) ]$
0. Min-max optimization. 

> TODO: this is incomplete. Read Min-max optimization