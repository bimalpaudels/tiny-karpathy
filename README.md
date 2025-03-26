## 🚀 Tiny Karpathy: Pretraining GPT on Karpathy’s Deep Learning Lectures
What happens when you train a GPT model on Andrej Karpathy’s own deep learning lectures? You get Tiny Karpathy—a character-level language model trained on transcripts from his YouTube series. Inspired by his Tiny Shakespeare example, 
I took it up a notch and created my own dataset to generate Karpathy-esque AI ramblings.

## Dataset
I copied and lightly formatted all transcripts from Karpathy’s deep learning series, ensuring proper punctuation and paragraph structure. No fancy preprocessing.

## Model Architecture
This model follows the GPT-style transformer Karpathy explains in Let's build GPT: from scratch, in code, spelled out.

- Character-level GPT (no tokenization)

- 6 transformer blocks with multi-head self-attention

Trained on Kaggle’s dual-GPU setup (used DataParallel for multi-GPU training)

## Training
- Trained for 2500/5000 iterations (~15 min on Kaggle’s GPUs)

- Training loss dropped below 1, with a clear separation from validation loss

Stopped early to avoid overfitting

## Sample Output
Here’s a small snippet of what Tiny Karpathy generated:
```Okay so the tokens otherwise or is this little bit and exactly talling of the matrix multiplic plus and on humant this is
three and the actual transformers and we make the one one-dimensionalizative memorize this is one of the gpt2 term dires thrivative
by a single like prefectively single node optimize these lines and that's now how many the element of the way that is powed following
it seefitely and pytorch we're doing that we ent so the um problems will be low by d so that's then we have because of the chain
here before we go into the projectivations so on. guessive I can've train into previous convolution I can over Yask linearity to think
the letters we can looked some supeound of like a just find the loss B and it's a too instead torch mlpful brach with and so now we see
the inkild the that instead we would have to be a tund then then than you go to have to implemend in my the bring case on if you need
to an text of tensor in a biable how default are deceively so this madded.
```


Check out the complete implementation and generated text in the repository.
