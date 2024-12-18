# Neuro-Symbolic Concept Composer


## Setup
Run the following commands to setup NeSyCoCo.

Make a new conda environment.
```bash
  conda create -n nesycoco python=3.10
  conda activate nesycoco
```

Install [Jacinle](https://github.com/vacancy/Jacinle).
```bash
  git clone https://github.com/vacancy/Jacinle --recursive
  export PATH=<PATH_TO_JACINLE>/bin:$PATH
```

Install [Concepts](https://github.com/concepts-ai/concepts).
```bash
  git clone https://github.com/concepts-ai/Concepts.git
  cd Concepts
  pip install -e .
```

### Glove Langauge Encoder


To download the Glove word embeddings and save them at `word_embeds/glove`, you can follow these steps:

1. Open a terminal or command prompt.
2. Navigate to the directory where your NeSyCoCo project is located.
3. Run the following command to download the Glove embeddings:

```bash
wget http://nlp.stanford.edu/data/glove.6B.zip
```

4. Extract the downloaded zip file using the following command:

```bash
unzip glove.6B.zip
```

5. Move the extracted files to the desired location using the following command:

```bash
mv glove.6B.* NeSyCoCo/word_embeds/glove
```

After completing these steps, you should have the Glove word embeddings saved at `NeSyCoCo/word_embeds/glove` directory.


## Train & evaluation

Please see the individual READMEs to train and evaluate models. 

Install the following library to train models.
```bash
  conda install tensorflow
  pip install chardet
```


Note: Before running train & eval commands, set `export PATH=<PATH_TO_JACINLE>/bin:$PATH`.

## Warning
NeSyCoCo leverages a pre-trained large language model as its language interpreter, and hence, even though our prompts are general examples of first-order logic, we do not have direct control over the LLM's generation. The LLM may output harmful biases.

This code is based on the [LEFT framework](https://github.com/joyhsu0504/LEFT/tree/main).
