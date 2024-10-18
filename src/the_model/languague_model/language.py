#Language Model Structure
#The Language Model brings together all the layers discussed so far: Token Embeddings, Positional Encoding, Layer Normalization, and the Decoder Stack.
#
#1. Token Embeddings:
#The model first creates token embeddings. Each token, word, subword, or character in the input sequence is represented by a high-dimensional vector. This vector is initialized randomly and learned during training. The embedding dimension is an adjustable hyperparameter.
#
#2. Positional Encoding:
#Next, the model applies positional encoding to the token embeddings. This helps the model understand the order of the tokens in the sequence, which is crucial for many natural language tasks. In GPT, positional encoding is achieved by adding a vector to each token embedding, which encodes the token’s position in the sequence.
#
#3. Layer Normalization:
#After that, the model applies Layer Normalization to the result of the positional encoding. This normalization process adjusts the vectors so that each dimension has a mean of 0 and a standard deviation of 1, equalizing the influence of each dimension and helping to stabilize the learning process.
#
#4. Decoder Stack:
#The normalized output is then passed into a stack of decoders. Each decoder block in the stack consists of a self-attention mechanism and a feed-forward neural network, with layer normalization and residual connections applied throughout.
#
#After the data passes through the decoder stack, we reach the final part of the model: the Language Model Head.
#
#Language Model Head (LMHead)
#The Language Model class concludes with the LMHead layer. This layer is essentially a linear transformation that maps the high-dimensional output of the decoder stack back to the dimension of the token vocabulary.
#
#Simplified Example:
#Imagine you have a vocabulary of 10,000 unique words, and your decoder output dimension is 512. Each word in your vocabulary will have a unique 512-dimensional vector representation after passing through the decoder. But we want to assign a probability to each word in the vocabulary given the preceding context. This is where the LMHead comes in.
#
#The LMHead maps the 512-dimensional vector back to a 10,000-dimensional space, essentially assigning a score to each of the 10,000 words. These scores are then passed through a softmax function to convert them into probabilities.
#
#How the LMHead Works
#The LMHead is implemented as a subclass and uses a linear layer (a basic layer that applies a linear transformation) to map the decoder's output dimension to the number of tokens in the vocabulary.
#
#For example, if we have a vocabulary of 5 words, the LMHead will map the 512-dimensional vectors to a 5-dimensional vector. Each entry in this output vector corresponds to a word in the vocabulary and can be interpreted as a raw score for how likely that word is to follow the phrase.
#
#Example of Probabilities:
#Suppose the probabilities after applying softmax are:
#
#["jumps": 0.6, "sleeps": 0.05, "eats": 0.2, "runs": 0.13, "fox": 0.02].
#Thus, the model predicts "The quick brown fox jumps" as the most likely continuation of the input sequence.

from src.the_model.languague_model.encode_decode.encodes.position_encode import PositionalEncoding
from src.the_model.languague_model.encode_decode.embed.embedding import TokenEmbedding
from src.the_model.languague_model.encode_decode.decodes.decode import DecoderStack
from typing import Optional
import torch

class LMHead(torch.nn.Module):
    """
    Pytorch module for the language model head.
    The language model head is a linear layer that maps the embedding dimension to the vocabulary size.
    """

    def __init__(self, embedding_dimension: int, number_of_tokens: int):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_tokens = number_of_tokens
        self.linear = torch.nn.Linear(embedding_dimension, number_of_tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the language model head.

        x dimensions are: (batch_size, sequence_length, embedding_dimension)
        output dimensions are: (batch_size, sequence_length, number_of_tokens)
        """
        # Compute the linear layer
        # linear_output dimensions are: (batch_size, sequence_length, number_of_tokens)
        linear_output = self.linear(x)

        return linear_output


class LanguageModel(torch.nn.Module):
    
    def __init__(
            self,
            cuda:                   torch.device,
            number_of_tokens:       int,                                # The number of tokens in the vocabulary
            max_sequence_length:    int                         = 512,  # The maximum sequence length to use for attention
            embedding_dimension:    int                         = 512,  # The dimension of the token embeddings
            number_of_layers:       int                         = 6,    # The number of decoder layers to use
            number_of_heads:        int                         = 4,    # The number of attention heads to use
            feed_forward_dimension: Optional[int]               = None, # The dimension of the feed forward layer
            dropout_rate:           float                       = 0.1   # The dropout rate to use
    ):
        super().__init__()
        self.number_of_tokens = number_of_tokens
        self.max_sequence_length = max_sequence_length
        self.embedding_dimension = embedding_dimension
        self.number_of_layers = number_of_layers
        self.number_of_heads = number_of_heads
        
        if feed_forward_dimension is None:
            # GPT-3 paper uses 4 * embedding_dimensons for the feed forward dimension
            self.feed_forward_dimension = embedding_dimension * 4
        else:
            self.feed_forward_dimension = feed_forward_dimension

        self.dropout_rate = dropout_rate

        # Create the token embedding layer
        
        self.token_embedding = TokenEmbedding(embedding_dimension, number_of_tokens)

        # Create the positional encoding layer
        self.positional_encoding = PositionalEncoding(embedding_dimension, max_sequence_length, cuda)

        # Create the normalization layer
        self.layer_normalization = torch.nn.LayerNorm(embedding_dimension)

        # Create the Decoder Stack
        self.decoder = DecoderStack(
            embedding_dimension=embedding_dimension,
            number_of_layers=number_of_layers,
            number_of_heads=number_of_heads,
            feed_forward_dimension=self.feed_forward_dimension,
            dropout_rate=dropout_rate,
            max_sequence_length=max_sequence_length
        )

        # Create the language model head
        self.lm_head = LMHead(embedding_dimension, number_of_tokens)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # compute the token embeddings
        # token_embeddings dimensions (batch_size, sequence_length, embedding_dimension)

        token_embeddings = self.token_embedding(x)
    
        # Compute the positional encoding
        # positional_encoding dimensins: (batch_size, sequence_length, embedding_dimension)
        positional_encoding = self.positional_encoding(token_embeddings)
    
        # post embedding layer normalization
        positional_encoding_normalized = self.layer_normalization(positional_encoding)
        decoder_outputs = self.decoder(positional_encoding_normalized, mask)
        lm_head_outputs = self.lm_head(decoder_outputs)
    
        return lm_head_outputs
        
    # Saving the model:
    def save_checkpoint(self, path: str) -> None:
        torch.save({
            'number_of_tokens': self.number_of_tokens,
            'max_sequence_length': self.max_sequence_length,
            'embedding_dimension': self.embedding_dimension,
            'number_of_layers': self.number_of_layers,
            'number_of_heads': self.number_of_heads,
            'feed_forward_dimension': self.feed_forward_dimension,
            'dropout_rate': self.dropout_rate,
            'model_state_dict': self.state_dict()
        }, path)
        
    @staticmethod
    def load_checkpoint(path: str,cuda: torch.device) -> 'LanguageModel':
        
        checkpoint = torch.load(path)
        
        model = LanguageModel(
            cuda,
            number_of_tokens=checkpoint['number_of_tokens'],
            max_sequence_length=checkpoint['max_sequence_length'],
            embedding_dimension=checkpoint['embedding_dimension'],
            number_of_layers=checkpoint['number_of_layers'],
            number_of_heads=checkpoint['number_of_heads'],
            feed_forward_dimension=checkpoint['feed_forward_dimension'],
            dropout_rate=checkpoint['dropout_rate']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model.to(cuda)