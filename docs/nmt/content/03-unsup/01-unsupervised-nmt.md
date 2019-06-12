Un Supervised NMT
===================

Training NMT models without using a single parallel data sounds like a daunting challenge at first - how is it even possible?
There has been surprising amount of progress made on this challenge recently (2017-2019).
There were efforts to train Statistical MT systems without using bitext, notably the paper titled [Deciphering foreign language](https://www.aclweb.org/anthology/P11-1002) {% cite ravi2011deciphering %}.
When it comes to NMT, we see a natural progression of the task: first, learn word translations without any data, followed by a way to take advanatge of them for to sentence translations.

Unsupervised Word alignments, done in embedding space, is generally a two step process:
1. Learn monolingual embeddings seperately
2. Learn the transformation function mapping one space to another.

There are plenty of ways to learn word embeddings: Word2vec, Fasttext, Glove, etc Each have their own advantages (and some drawbacks). The task is straight forward application of distributional hypothesis on monolingual data.
The second step, learning transformation function for mapping one to another is some what interesting for translation community.

Progression of learning word embedding alignments:
1. Using a dictionary of word translations to learn the transformation matrix. [Exploiting Similarities among Languages for Machine Translation](https://arxiv.org/pdf/1309.4168.pdf) {% cite mikolov2013exploiting %}
2. Much smaller dictionaries: as small as 25 pairs. In many cases those pairs can be automatically obtained, eg: numbers, names etc. See [Learning bilingual word embeddings with (almost) no bilingual data](https://www.aclweb.org/anthology/P17-1042) {% cite artetxe2017binlingemb %}
3. More fancier techniques for learning word alignments: Use adversarial training. See [Word translation without parallel data](https://arxiv.org/pdf/1710.04087.pdf) {% cite conneau2017word %} .

The implementation of {% cite conneau2017word %} 's approch is made available on github as [facebookresearch/MUSE](https://github.com/facebookresearch/MUSE) and it is more popular. These unsupervised approaches can sometimes perform better than supervised approach. Here is a visual interepretation of embedding alignment (taken from their github repo): ![](https://github.com/facebookresearch/MUSE/raw/master/outline_all.png)



In summary, unsupervised word alignments exploited these two phenomenons:
1. Words having similar meaning appear in similar context across languages
2. There is a linear mapping from one embedding vector space to another which can be easily learned

Once we had these automatically aligned word embeddings of good enough quality, the next problem to tackle was: "how to do sentence translation without parallel data?"


### 1. [Unsupervised neural machine translation](https://arxiv.org/pdf/1710.11041.pdf) {% cite artetxe2017unsupervised%}

+ Crosslingual embeddings: Skipgram word2vec, 10 neg sample, 10 ctx window, 300 dims. Then aligned using {% cite artetxe2017binlingemb %}
+ NMT Architecture: 2 layer Bi-GRU encoder, 2 layers GRU dec, 600 hid dim, 300 dim embs, general attention
  + Both directions: Source -> Target and Target -> Source
  + Shared encoder for both source and target, decoders are seperate. Encoder embeddings are fixed. Decoders are let to evolve.
  + Vocabularies are separate for both languages. So embedding matrices are seperate (but aligned)
+ Denoising: for a seq of $N$ toks, $N/2$ swaps are performed as nimitation of oise. Without noise, copy task is too trivial, AEs doesnt learn any useful representation.
+ On-the-fly back translation: they use the model to generate translation (in the inference mode with greedy dec) then reconstruct from the translation. Backtranslation is much noisier form than random word swaps.
+ Switch the training steps: between Denoise L1 to L1; Denoise L2 to L2; Cycle via BackTranslation: L1 -> L2 -> L1; Cycle via BackTranslation: L2 -> L1 -> L2.
+ Trained on short seqs: 50 or fewer toks (after BPE)
+ Crosslingual embeddings:
+ Took 4-5 days to train. Results on WMT: Fr-en 15.6; De-en 10.2

### 2. [Unsupervised machine translation using monolingual corpora only](https://arxiv.org/pdf/1711.00043.pdf) {% cite lample2017unsupervised %}

* Cross lingual embeddings from {% cite conneau2017word %} i.e. MUSE.
* Learn to reconstruct in both langs from the shared latent space
* Denoising auto encoder, reconstruct from a noisy input
* (This paper has good citations to relevant work; semi supervised, autoencoder etc)

* LSTM with 300 dims, 3 layers. Two models: Src-to-tgt and tgt-to-src models: all encoder layers are shared, all decoder layers are shared. So only generator matrix is different for both the languages. They actually have two models, LSTM layers are shared.
* Denoising auto encoders: drop words, shuffle the order of tokens (upper bound k on how many timesteps max a token can move). They found 10% word drop and k=3 be good parameters
* Cross-domain loss (aka cycle loss) x -> C(x) -> y -> C(y) -> x'.     C is corruption operator. Similarly, y -> C(y) -> x -> C(x) -> y'.
* Adversarial training: Encoding of x and y sentences should be indistinguishable
* Final objective is: is linear combination of following
  * AE src->src
  * AE tgt->tgt
  * CycleLoss src->tgt->src
  * CycleLoss tgt->src->src
  * Adversarial Loss
* Results On WMT fr-en 14.3 while supervised is 26.1; de-en is 13.3 while supervvised is 25.6


### 3. [Unsupervised neural machine translation with weight sharing](https://arxiv.org/pdf/1804.09057.pdf) {% cite yang2018unsupervised %}

+ Artetxe et al 2017 use a shared encoder; but seperate decoders. Lample et al 2017 share a single encoder and single decoder (fully shared).
+ Conjecture: Fully shared is not  good, since languages have different syntax; completely different is also not good, since forcing common latent space is hard. We want to something in the middle: partial sharing.
+ Encoder: share the higher layers; allow the lower layers (close to embeddings) to be different
+ Decoder share the lower layers; allow the higher layers (close to generator module) to be different
  + Question: Decoder too has input embedding too just like encoder, but they are okay shared? Maybe: since word embeddings are in common latent space
+ Two kinds of adversarial discriminators are built in:
  + Prior work (Yang et all 2017, same first author): "Improving Neural Machine Translation with Conditional Sequence Generative Adversarial Nets"
  + Loss is modified to include regular loss + GAN reward
  + Two Discriminators: Local (encoded repr of source & target are indistinguishable), Global (model generated sentences are indistinguishable from human generated sentences)
+ Slight improvement in BLEU (+1.0) compared to Lample et al 2017

### 4. [Phrase-Based & Neural Unsupervised Machine Translation](https://www.aclweb.org/anthology/D18-1549){% cite lample2018unsup-mt %}

+ Simplification from lample et al 2017 and Artetxe et al 2017. In essence these three work for unsupervised MT (ablation study proved it):
  + Initialization of embeddings via crosslingual embeddings
  + strong language models via denoising autoencoder
  + automatic backtranslation
+ They train NMT and PBSMT (moses). Source code at [facebookresearch/UnsupervisedMT](https://github.com/facebookresearch/UnsupervisedMT)
+ NMT training
  + two varieties: LSTM (told that the setup is same as Artetxe et al 2017 but they had used GRUs!), Transformer (4 layers); 512 dim.
  + Shared Encoder, forced interlingua using adversarial loss. Shared Decoder too for regularization
  + BOS (first token) of the shared decoder is the language id token. Encoder's first token is not modified unlike Johnson et al 2016.
  + Loss is Denoising reconstruction + backtranslation loss ; Note: No adversarial loss ? Maybe bcoz the encoder is shared, no need to explicitly force it to learn interlingua
+  NOTE: authors never explicitly mentioned that the weights of 3 out of 4 layers were shared as suggested by Yang et al 2018 (see above). However in the code it is done. Code didn’t run our cluster and authors said to use XLM instead which is much better (see below) (Lample & Artetxe, 2019)
+ Validation and model selection via roundtrip BLEU ( x -> y -> x;  y-> x->y).
+ Note: Roundtrip BLEU correlates well for transformer but not for LSTMs, nobody knows why! They used a small validation set of 100 parallel sentences for LSTMs validation

### [Cross-lingual Language Model Pretraining](https://arxiv.org/pdf/1901.07291.pdf) {% cite DBLP:journals/corr/abs-1901-07291 %}
+ Code is made available on gitub at [facebookresearch/XLM](https://github.com/facebookresearch/XLM) . (Very well written!)
+ Improvements over lample et al 2018; better initialization
+ Instead of  initializing (just) embeddings they train auto encoders with different objectives: causal LM (predict next token given left context); masked LM predict some masked token in sequence given both left and right context
+ If you have parallel data, you can do Translation LM; concat src + tgt sequence, mask out some sequences on both source and target! Nice!! (we should try this way of training transformer model)
+ Mix all the languages; adjust sampling to balance low and high resources
+ Denoising AE and online backtranslation are still the key
+ High BLEU scores comparable with supervised MT (going to try this)
+ Computationally expensive: Language models are way expensive (they needed 64 top of the class GPUs), MT finetuning is relatively inexpensive however still expensive (they needed 8 GPUs)

## References

{% bibliography --cited %}
