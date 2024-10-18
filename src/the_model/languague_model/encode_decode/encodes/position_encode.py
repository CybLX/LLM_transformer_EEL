#Positional Encoding in Sequence Models
#Positional encoding is essential for enabling models to understand the order of elements in a sequence. This is achieved by creating a vector for each position in the sequence and adding it to the corresponding input vector. By doing so, the model can learn to utilize these position vectors effectively.
#Calculation of Positional Encoding
#The positional encoding for a given position p and dimension 𝑖 in the input sequence is computed using sine and cosine functions as follows:
#For even dimensions:
# PE(p, 2i) = sin( p / 10000^(2i/d_model))
#For odd dimensions:
# PE(p, 2i+1) = cos(p / 10000^(2i/d_model))
#
# In these equations,𝑑_model represents the dimensionality of the input and output vectors. 
# A larger d_model implies that each word is represented by a higher-dimensional vector, allowing for more complex representations. However, this also requires more computational resources.
#
#Characteristics of Positional Encoding
#The sine and cosine functions are applied alternately to each dimension of the positional encoding vector, creating a unique pattern that the model can learn and recognize. The resulting positional encoding values range between -1 and 1 and exhibit a wavelength that increases with each dimension. This pattern helps the model distinguish different positions within the sequence and generalize to sequence lengths not encountered during training.
#
#Integration with Input Embeddings
#The positional encodings are added to the input embeddings before being processed by the transformer model. This integration allows the transformer to leverage the positional information effectively, adapting it to suit the specific task at hand.

import numpy as np
import torch

class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, max_sequence_length: int, cuda: torch.device):
        super().__init__()

        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        self.positional_encoding = self.create_positional_encoding(cuda)
    
    def create_positional_encoding(self,cuda: torch.device) -> torch.Tensor:
        # Initialize encoding matrix
        positional_encoding = np.zeros((self.max_sequence_length, self.d_model))

        # calculate positional encoding for each position and each dimension
        for pos in range(self.max_sequence_length):
            for i in range(0, self.d_model, 2):
                #apply sin to even indices in the array
                # indicies in python start at 0 so is is even:
                positional_encoding[pos, i] = np.sin(pos / (10000 ** ((2*i) / self.d_model)))

                if i + 1 <self.d_model:
                    positional_encoding[pos, i+1] = np.cos(pos/ (10000 ** ((2*i) / self.d_model)))

        return torch.from_numpy(positional_encoding).float().to(cuda)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add positional encoding to input embeddings. The ":" indexing ensures we only add positional encodings to the length of the sequence in the batch.
        # x.size(0) is the batch size, so this is a way to make sure we're not adding extra positional encodings.
        return x + self.positional_encoding[:x.size(1), :]
    

