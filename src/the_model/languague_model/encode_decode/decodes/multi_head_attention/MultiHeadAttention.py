#Attention Mechanism Overview
#When computing attention scores, it is crucial to exclude the effects of padding tokens. To achieve this, a mask is applied to the attention scores matrix, effectively setting the scores at padding positions to a very large negative number (e.g., -1e9). This approach ensures that when the scores are passed through a softmax function, they transform into zeros, meaning that padding positions will not influence the final output of the attention layer.
#
#Masking Mechanism
#The masking operation is applied to the attention_weights tensor. In this operation, positions corresponding to a zero in the mask are assigned a value of -1e9. This prevents the model from considering future information during predictions, adhering to the autoregressive property where the prediction at a given time step depends solely on the inputs and outputs from previous time steps.
#
#To reinforce this property during model training, a subsequent mask is utilized. This mask ensures that for any given word, the model cannot attend to the words that follow it, preserving the integrity of the prediction process.
#
#Attention Scores Calculation
#Attention scores serve to highlight the important parts of an input sequence, similar to how humans focus on specific segments of a sentence or image based on context. These scores are derived using query, key, and value vectors, created by multiplying the input embeddings with learned matrices (essentially a linear transformation).
#
#The attention scores for the tokens are calculated as the dot product between the query and key vectors. To enhance gradient stability, these scores are scaled down by dividing by the square root of the dimension of the query/key/value vectors.
#
#Masking Attention Scores
#It is important to note that attention can be masked. For masked positions, the attention score is set to negative infinity, effectively rendering these positions non-existent in subsequent computations.
#
#Finally, the attention scores are normalized using the softmax function, ensuring that they fall between 0 and 1 and sum to 1. The normalized attention scores are then multiplied by the value vectors and summed to produce the output of the attention layer.

# Basic attention layers:
import numpy as np
import torch

class MaskedSelfAttention(torch.nn.Module):

    def __init__(self,embedding_dimension: int, head_dimension: int):
        super().__init__()

        self.embedding_dimension = embedding_dimension
        self.head_dimension = head_dimension
        
        self.query_layer    = torch.nn.Linear(embedding_dimension, self.head_dimension)
        self.key_layer      = torch.nn.Linear(embedding_dimension, self.head_dimension)
        self.value_layer    = torch.nn.Linear(embedding_dimension, self.head_dimension) 

        self.softmax = torch.nn.Softmax(dim = -1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        # x dimensios are: (batch_size, sequence_length, embeddiing_dimension)
        # query, key, value are: (batch_size, sequence_length, head_dimension)
        query = self.query_layer(x)
        key = self.key_layer(x)
        value = self.value_layer(x)

        # Calculate the attentiong Weights:
        # Attention_weights dimensions are: (batch_size, sequence_lenght, sequence_length)

        attention_weights = torch.matmul(query, key.transpose(-2, -1))

        #Scale the attention weights:
        attention_weights = attention_weights/ np.sqrt(self.head_dimension)
        
        # Aply the mask to the attention weights, by setting the masked tokens to a veru low value.
        # this will make the softmax output 0 for these values.
        # Apply yhe mask to the attention weights.
        mask = mask.reshape(attention_weights.shape[0], 1, attention_weights.shape[2])
        attention_weights = attention_weights.masked_fill(mask == 0, -1e9)

        # Sfotmax make sure all scores are between 0 and 1, and the sum of them is 1.
        # attention_scores dimensions are: (batch_size,sequence_length, sequence_length)

        attention_scores = self.softmax(attention_weights)

        # The attention scores are multiplied by the value
        # Values of tokens with high attention score get highlighted because they are multiplied by a larger number
        # and tokens with low attention scores get downed out because theu are muultiplied by a smaller number

        #output dimensions are (batch_size, sequence_length, head_dimension)

        return torch.bmm(attention_scores, value)
    


# The attention mechanism allows a model to focus on different parts of the input, sequence when generating each word in the output sequence.
# Instead of having one single attention perspective, why not have multiple?

# Eache head computes its own query, key, value matrices from the input.
# They then compute their attention scores and produce an output. But instead of being processed individually, 
# the output of each head is concatenated toget and the transformed with a linear layer to roduce the final output.


class MaskedMultiHeadedSelfAttention(torch.nn.Module):

    def __init__(self, embedding_dimension: int, number_of_heads: int):
        super().__init__()

        self.embedding_dimension = embedding_dimension
        self.head_dimension = embedding_dimension // number_of_heads
        self.number_of_heads = number_of_heads

        #create the self attention modules.
        self.self_attentions = torch.nn.ModuleList(
            [MaskedSelfAttention(embedding_dimension, self.head_dimension) for _ in range(number_of_heads)]
        )

        # create a linear layer to combine the outputs of the self attention modules
        self.output_layer = torch.nn.Linear(number_of_heads * self.head_dimension, embedding_dimension)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # compute the self attention for each head
        # self_attention_outputs dimensions: (number_of_heads, batch_size, sequence_length, head_dimension)
        self_attention_outputs = [self_attention(x,mask) for self_attention in self.self_attentions]

        # concatenate the self attentions outputs
        # self_attention_outputs_concatenated dimensions: (batch_size, sequence_length, number_of_head * head_dimension)
        concatenated_self_attention_outputs = torch.cat(self_attention_outputs, dim=2)

        # Apply the output layer to the concatenated self attention outputs
        # output_layer dimensions: (batch_size, sequence_length, embedding_dimension)

        return self.output_layer(concatenated_self_attention_outputs)
    
