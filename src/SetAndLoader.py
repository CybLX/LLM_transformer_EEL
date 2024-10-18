from src.vocab_and_tokenize.tokenizer import vocabulario
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple
from tqdm import tqdm
import pandas as pd
import torch
import nltk

class DataFrameDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> pd.Series:
        # Aqui você pode adicionar pré-processamento adicional se necessário
        text = self.dataframe.iloc[idx]
        return text
    
class CommentsDataset(Dataset):
    def __init__(self, data_subset: DataFrameDataset, tokenizer: vocabulario, max_sequence_length: int):
        """
        data_subset: Um subset de um DataFrame que contém os comentários.
        tokenizer: O tokenizador que possui funções de tokenização, índices de padding e tokens especiais.
        max_sequence_length: O tamanho máximo da sequência.
        """
        self.data = data_subset  # O subset do DataFrame (ou série de comentários)
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.tokenized_data = self.tokenize_and_pad_data()

    def tokenize_and_pad_data(self) -> List[List[int]]:
        """
        Tokeniza e aplica padding nos comentários. Gera n-grams de tamanho max_sequence_length + 1.
        """
        flatten = lambda outer_list: [item for inner_list in outer_list for item in inner_list]

        # Gera n-grams e faz padding para cada sentença
        return flatten([
            list(nltk.ngrams(
                [self.tokenizer.pad_index] * self.max_sequence_length +  # Padding no início
                [self.tokenizer.start_index] +                           # Token de início
                self.tokenizer.tokenize(sentence) +                      # Tokens do comentário
                [self.tokenizer.end_index] +                             # Token de finalização
                [self.tokenizer.pad_index] * (self.max_sequence_length - len(self.tokenizer.tokenize(sentence)) - 2),  # Padding no final
                self.max_sequence_length + 1                             # Tamanho do n-grama
            ))
            for sentence in tqdm(self.data)  # Itera sobre os comentários no subset
        ])

    def create_mask(self, sequence_tensor: torch.Tensor) -> torch.Tensor:
        """
        Cria a máscara: 1 para tokens que não são <pad>, 0 para tokens <pad>.
        """
        mask_tensor = torch.ones_like(sequence_tensor)
        mask_tensor[sequence_tensor == self.tokenizer.pad_index] = 0
        return mask_tensor

    def __len__(self) -> int:
        """
        Retorna o número de sequências tokenizadas no dataset.
        """
        return len(self.tokenized_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retorna uma sequência e sua máscara correspondente no índice fornecido.
        """
        sequence = self.tokenized_data[idx]
        sequence_tensor = torch.tensor(sequence, dtype=torch.long)
        mask_tensor = self.create_mask(sequence_tensor)
        return sequence_tensor, mask_tensor

    def create_dataloader(self, batch_size: int, shuffle: bool = True, drop_last: bool = True) -> DataLoader:
        """
        Retorna o DataLoader configurado para esse dataset.
        """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=my_collate_fn)
    
def my_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Função para lidar com sequências de comprimento fixo e criar batches.
    """
    sequences, masks = zip(*batch)
    
    # Assegure que todos os tensores de entrada e máscara tenham o mesmo comprimento
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value = 1)
    padded_masks = pad_sequence(masks, batch_first=True, padding_value=0)
    
    return padded_sequences, padded_masks