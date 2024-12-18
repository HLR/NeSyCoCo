import torch


class WordEmbedding:
    def __init__(self):
        self.model_name = None
    
    @property
    def embedding_dim(self):
        raise NotImplementedError
    
    def get_embedding(self, text: str) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def get_words(text):
        if "_" in text:
            text = text.replace("_", " ")
        words = text.split()
        return words


class OneHotWordEmbedding(WordEmbedding):
    def __init__(self, max_size: int = 256):
        self.vocab = []
        self.max_size = max_size
        
    @property
    def embedding_dim(self):
        return self.max_size
    
    def get_embedding(self, text: str) -> torch.Tensor:
        embedding = torch.zeros(self.max_size)
        words = self.get_words(text)
        for word in words:
            if word not in self.vocab:
                self.vocab.append(word)
            if len(self.vocab) >= self.max_size:
                raise ValueError("Max vocab size exceeded")
            embedding[self.vocab.index(word)] = 1
        return embedding


class SpacyWordEmbedding(WordEmbedding):
    def __init__(self, model_name: str = "en_core_web_sm"):
        import spacy
        self.model = spacy.load(model_name)
        self._embedding_dim = 0
    
    @property
    def embedding_dim(self):
        if self._embedding_dim == 0:
            self._embedding_dim = self.model("hello").vector.shape[0]
        return self._embedding_dim
    
    def get_embedding(self, text: str) -> torch.Tensor:
        return torch.tensor(self.model(text).vector)

class RoBERTaWordEmbedding(WordEmbedding):
    def __init__(self):
        from transformers import RobertaModel, RobertaTokenizer
        self.model_name = "roberta-base"
        self.model = RobertaModel.from_pretrained(self.model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
    
    @property
    def embedding_dim(self):
        return self.model.config.hidden_size
    
    def get_embedding(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            output = self.model(**tokens)
        return output.last_hidden_state.mean(dim=1).squeeze()


class FastTextWordEmbedding(WordEmbedding):
    def __init__(self):
        import fasttext
        fname = "word_embeds/fasttext/dbpedia.bin"
        self.model = fasttext.load_model(fname)
    
    @property
    def embedding_dim(self):
        return self.model.get_dimension()
    
    def get_embedding(self, text: str) -> torch.Tensor:
        return torch.tensor(self.model.get_sentence_vector(text))
    
class GloveWordEmbedding(WordEmbedding):
    def __init__(self):
        from gensim.models import KeyedVectors
        fname = "word_embeds/glove/glove.6B.300d.txt"
        self.model = KeyedVectors.load_word2vec_format(fname, binary=False,no_header=True)
        
    @property
    def embedding_dim(self):
        return self.model.vector_size
    
    def get_embedding(self, text: str) -> torch.Tensor:
        ### Extract all the words from the text if it is a sentence
        words = self.get_words(text)
        embedding = torch.zeros(self.embedding_dim)
        ### Iterate over all the words and add their embeddings
        for word in words:
            if word in self.model:
                embedding += torch.tensor(self.model[word])
        return embedding
    