#Decoder Layer Overview
#Each decoder layer consists of a self-attention mechanism and a feed-forward neural network. The self-attention mechanism enables the model to weigh the importance of different tokens when predicting the next token, while the feed-forward network learns traditional positional relationships between tokens. Multiple decoder layers are stacked to capture complex patterns and deeper contextual relationships.
#
#Step-by-Step Process
#Step 1: Input and Layer Normalization
#When a decoder receives its input, it first applies layer normalization. This process is crucial for numerical stability, preventing the problem of exploding gradients by keeping values within a manageable range. Layer normalization also ensures that each feature has a mean of 0 and a standard deviation of 1, balancing the influence of all dimensions for efficient learning.
#
#Step 2: Self-Attention
#After normalization, the decoder layer applies the self-attention mechanism. This allows the model to assess the importance of each token in the sequence relative to others, creating context-aware representations.
#
#Step 3: Residual Connections
#The output from the self-attention process is added back to the original normalized input through residual connections. This technique helps gradients flow more effectively through the network, mitigating the vanishing gradients problem, which occurs when gradients become too small during backpropagation.
#
#Step 4: Second Normalization and Feed-Forward Network
#Following the residual connection, the output is normalized again and passed through a feed-forward neural network (FFNN). This fully connected layer applies non-linear transformations, such as ReLU, enabling the model to learn complex relationships in the data.
#
#Step 5: Dropout
#To prevent overfitting, a dropout procedure is applied, randomly removing a percentage of connections (e.g., 10%) during training. This encourages the model to develop robust representations that do not depend heavily on specific connections.
#
#Output Propagation
#After passing through these steps, the output of one decoder layer becomes the input for the next. This stacking of multiple decoder layers allows the model to learn representations at various levels of abstraction, essential for understanding and generating human-like text.
#
#Structural Components
#The DecoderLayer encapsulates the process within a single layer, while the Decoder Stack manages the stacking of multiple decoder layers.

from src.the_model.languague_model.encode_decode.decodes.multi_head_attention.MultiHeadAttention import MaskedMultiHeadedSelfAttention
import torch



class FeedForward(torch.nn.Module):

    def __init__(self, embedding_dimension: int, feed_forward_dimension: int):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.feed_forward_dimension = feed_forward_dimension
        
        self.linear_1 = torch.nn.Linear(embedding_dimension, feed_forward_dimension)
        self.linear_2 = torch.nn.Linear(feed_forward_dimension, embedding_dimension)

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(
                    torch.relu(
                        self.linear_1(x))) 




class DecoderLayer(torch.nn.Module):

    def __init__(self,embedding_dimension: int, number_of_heads: int,feed_forward_dimension: int, dropout_rate: float):
        super().__init__()

        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.feed_forward_dimension = feed_forward_dimension
        self.dropout_rate = dropout_rate

        self.multi_headed_self_attention = MaskedMultiHeadedSelfAttention(embedding_dimension,number_of_heads)

        self.feed_forward = FeedForward(embedding_dimension, feed_forward_dimension)

        self.dropout = torch.nn.Dropout(dropout_rate)

        self.layer_normalization_1 = torch.nn.LayerNorm(embedding_dimension)
        self.layer_normalization_2 = torch.nn.LayerNorm(embedding_dimension)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        # STEP 1: layer normalization 1
        normalized_x = self.layer_normalization_1(x)

        # STEP 2: Multi Headed Self Attention
        attention_output = self.multi_headed_self_attention(normalized_x, mask)

        # STEP 3: Residual Output
        residual_output = x + attention_output

        # STEP 4: Layer normalization 2
        normalized_residual_output = self.layer_normalization_2(residual_output)
        # Feed Forward
        feed_forward_output = self.feed_forward(normalized_residual_output)

        # STEP 5: Dropout, only when training,
        if self.training:
            feed_forward_output = self.dropout(feed_forward_output)
        
        # residual output
        return residual_output + feed_forward_output
    

class DecoderStack(torch.nn.Module):

    def __init__(self, embedding_dimension: int, number_of_layers: int, number_of_heads: int, feed_forward_dimension: int, dropout_rate: float, max_sequence_length: int):
        super().__init__()

        self.embeffing_dimension = embedding_dimension
        self.number_of_layers = number_of_layers
        self.number_of_heades = number_of_heads
        self.feed_forward_dimension = feed_forward_dimension
        self.dropout_rate = dropout_rate
        self.max_sequence_length = max_sequence_length

        # Create the encoder Layers
        self.encoder_layers = torch.nn.ModuleList(
            [DecoderLayer(embedding_dimension, number_of_heads, feed_forward_dimension, dropout_rate)
             for _ in range(number_of_layers)]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        decoder_outputs  = x
        for decoder_layer in self.encoder_layers:
            decoder_outputs = decoder_layer(decoder_outputs, mask)
        return decoder_outputs
    