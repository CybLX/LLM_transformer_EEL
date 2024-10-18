#Understanding Tokenization and Embeddings
#Tokens can be thought of as atomic units of meaning in a language. They can represent words, parts of words, or even entire phrases. During the tokenization process, these units are replaced with corresponding integers. However, simply using integers is not sufficient to capture the nuances of language and the relationships between different tokens.
#
#The Role of Embedding Layers
#This is where embedding layers come into play, acting as a bridge between discrete tokens and continuous vector space. The embedding layer transforms each token, represented as an integer, into a continuous vector in a high-dimensional space.
#
#Implementing Embeddings in PyTorch
#In PyTorch, the nn.Embedding class is utilized to create the embedding layer. During training, the weights of this layer—representing our token embeddings—are updated and fine-tuned to better capture the semantic relationships between different tokens.
#
#Key Parameters
#number_of_tokens: This parameter indicates the total number of unique tokens that the model may encounter in the input. It typically equals the size of the token dictionary.
#
#d_model: This parameter specifies the size (dimensionality) of the embedding vectors. Higher dimensions allow for encoding more information about each token, but they also increase the computational complexity and the time required to train the model.
#
#Input and Output of the TokenEmbedding Module
#The input to our TokenEmbedding module is a batch of sequences, where each token is represented by an integer. In contrast, the output is a batch of the same sequences, but each integer is replaced by a rich, high-dimensional vector that encapsulates semantic information.

import torch

class TokenEmbedding(torch.nn.Module):
    def __init__(self,d_model: int, number_of_tokens: int):
        super().__init__()

        self.embedding_layer = torch.nn.Embedding(
            num_embeddings= number_of_tokens,   # Total number of unique tokens the model can encouter, typically equals the size of dictionary
            embedding_dim= d_model              # Size of embedding vectors, more higher the dimensions, more encoded information about each token, more computational complexity
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding_layer(x)
    
