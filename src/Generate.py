from src.vocab_and_tokenize.tokenizer import vocabulario
from typing import Optional
import torch

def pad_left(sequence: list[int], final_length: int, padding_token: int) -> list[int]:
    return [padding_token] * (final_length - len(sequence)) + sequence


class Generator:

    def __init__(self, model: torch.nn.Module, tokenizer: vocabulario):

        self.model = model
        self.tokenizer = tokenizer

    def generate(self, device:              torch.device,
                 max_tokens_to_generate:    int,
                 prompt:                    Optional[str]   = None,
                 temperature:               float           = 1.0,
                 eos_token:                 int             = None,
                 padding_token:             int             = 0) -> str:
        
        self.model.eval()

        if prompt is None:
            start_tokens = [self.tokenizer.lookup_token(padding_token)]
        else:
            start_tokens = self.tokenizer.tokenize(prompt)
        

        # Generation input
        input_tensor = torch.tensor(
            pad_left(
                sequence=start_tokens,
                final_length=self.model.max_sequence_length + 1,
                padding_token=padding_token
            ),
            dtype=torch.long
        ).to(device)

        num_dims = len(input_tensor.shape)

        if num_dims == 1:
            input_tensor = input_tensor[None, : ]
        
        for _ in range(max_tokens_to_generate):

            x = input_tensor[:, -self.model.max_sequence_length:]

            mask = torch.ones_like(x)
            mask[x == padding_token] = 0

            # Compute the next token probabilities
            next_token_probabilities = self.model.next_token_probabilities(
                x = x,
                temperature = temperature,
                mask = mask
            )

            # Sample the next token from the probability distribution
            next_token = torch.multinomial(next_token_probabilities, num_samples = 1)

            # Append the next token to the output
            input_tensor = torch.cat([input_tensor, next_token], dim = 1)

            # If the end of sequence token is reached, stop generation tokens
            if eos_token is not None and next_token == eos_token:
                break

        generated_tokens = input_tensor[0].tolist()
        return ' '.join([self.tokenizer.lookup_index(token) for token in generated_tokens])
    

