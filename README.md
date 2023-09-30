# Mini N-gram Model Text Generation

This project was inspired from the bigram model from the tutorial [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2064s&ab_channel=AndrejKarpathy) by Andrej Karpathy. This code also is in PyTorch instead of TensorFlow unlike some of my other projects.

## How to Use

If you just want to try it out, there is an online demo [here](https://www.shivam.pro/projects/ngram).

First, in [ngram.ipynb](/ngram.ipynb), the whole model is trained in PyTorch and saved as an ONNX file along with the vocab. Next, in [onnx.ipynb](/onnx.ipynb), the saved model is loaded and used to confirm that it works.

## About the Model

This is my own implementation based on the bigram model, but configurable to use n previous tokens. The model has a second hyperparameter $n \ge 3$ which is the number of tokens the model will look at. First, it vectorizes the tokens using an embedding. Next, it concatenates all $n$ embeddings into a vector of size $n \ \times$ $`\text{embedding\_size}`$. Next, it is passed through an intermediate fully-connected linear layer followed by ReLU activation and $0.2$ dropout. Finally, it is then passed through a linear fully-connected layer to get the logits. This model is trained on the [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset, however it can be trained on any text dataset.

### Model Architecture:

```python
n = 7 # default value but can be changed
token_embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
fc = nn.Linear(embedding_size * n, intermediate_size)
dropout = nn.Dropout(0.2)
final = nn.Linear(intermediate_size, vocab_size)


input_tokens = ... # shape (n,)
x = token_embedding(input_tokens)
x = flatten(x) # shape (n * embedding_size,)
x = fc(x) # shape (intermediate_size,)
x = ReLU(x) # shape (intermediate_size,)
x = dropout(x) # shape (intermediate_size,)
x = final(x) # shape (vocab_size,)
```

## Sample Output

Sample output with input `"LUCENT"`, $n = 7$, and temperature $1.0$

```
LUCENTIO:
Whel queetord; ous beft sut up forstipu
Whil tife nin:
Aworld,
You that that spoter abrian un. Lity besialed, on come us lay this fold kiss bid,
What ratssmal: what!
Seray ur stowarn?

CESTER:
Mo.

GLOUCESTER:
Are dis,
That I no illome?

JULIET:
O liven in anoth my fallen mer, threnguest,
Thes difulf.
```

The model outputs plausible words, but the sentences don't make sense. This is because the model is only trained on a small dataset, and the model is not very complex. The model is also trained on a character level, so it doesn't know what words are. It only knows what characters are. It is still impressive for only 65K parameters and character level tokenization.

## Future Work

In my other repo [mini-gpt](https://github.com/shivamCode0/mini-gpt), I am working on more complex models that are more complex. This is just a simple baseline and proof of concept that demonstrates using PyTorch to create and train a model, and running it on the web using ONNX.
