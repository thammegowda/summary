Background
==========

**Auto Encoder** learns representations by the technique of compression (aka dim reduction) followed by decompression.
The compact representation is often called "code".
For an high demnsiomal data $X$, we define encoder as a parameterized function
$Enc_\phi : X \rightarrow F$  
Where $F$ is a compact representation in lower dimensions.
Simultaneously, we pair this compressing function with a decompressing function to reconstruct the original data.  
$Dec_\psi : F \rightarrow X$

Usually, these compressor and decompressors are lossy. Ideally we want the reconstruction to as close as possible to input (thats the learning objective of learning the parameters $\phi$ and $\psi$). The objective here is to minimize the reconstruction error, which can be expressed as $\Vert X − Dec_\psi(Enc_\phi(X))\Vert^2$.

However, this is still very abstract - we dont know how input $X$ is, how many parameters in $\phi$ and $\psi$, and how to set them. Usually, they are all depends on the input $X$.

We want to approximate these parameterized functions (i.e. approximate parameters  $\phi$ and $\psi$).
We define an optimizer that can directly adjust these parameters by utilizing a set of samples (training data). The chosen dataset, and the chosen optimization algorithm can have a noticable effect (or bias) in the estimation of parameters, which inturn affects the end objective. Another implicit objective is to be generic enough, i.e. if given another set of samples (test set), the parameters should still behave as desired.
From the theoretical point of view, this is maximum likelihood estimation (MLE) over a set of samples. In simple words, its the task of finding maximum-likelihood (i.e. most likely) values of $\phi$ and $\psi$ on a give set of samples (training set) while also keeping the estimation generic enough to be used on other unseen samples (test set).
In summary, we have two objectives:

+ **Objective 1:** Better reconstruction i.e. $Dec_\psi(Enc_\phi(X))$ should be very similar to input $X$
+ **Objective 2:** Better generalization: This method should also work on unseen samples.

Deep learning / Neural networks are one way of accomplishing these defined objectives. So $Enc_\phi$ and $Dec_\psi$ are two neural networks. The nice thing about networks is that we can connect two or more of them to form a bigger network. So Autoencoder in our context is a big network which has two sub networks - encoder and decoder.

Lets take images as our input, since they are simpler and visually inspectable. They can be resized to have fixed number of dimensions without losing much of information (after all, we are in the context of lossy compressions and approximations of those lossy compressions). The reason for chosing images over linguistic structures (scuh as sentences) is that - unlike images sentences cannot be easily resized to have fixed dimensions, and they have temporal property - so more complicated!

In the computer vision we use a kind of networks called convolusion neural networks (with some more fancier additions) to approximate $Enc_\phi$ and $Dec_\psi$.
Experimental evidance has shown that this kind of Encoding and Decoding easily satisfies Reconstruction objectives, however fails to satisfy generalization objective.

**Denoising Auto Encoder**
In this setup, the input $X$ is intentionally corrupted. The task of encoder is to reconstruct the clean input from noisy input. This task is some what challenging than the vanilla task, so forces the models to learn better representations.
