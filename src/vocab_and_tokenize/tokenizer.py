#%%
# Tokenization                  : Text to tokens
# Input Embeddings              : Tokens to Vectors
# Positional Encodings          : Retaining sequence order information
# Masking                       : Preventing model from  peeking ahead or looking at padding tokens
# Multi-headed self-attention   : Relating different input tokens
# Decoder Stack                 : Refining understanding over multiple layers
# Language Model Head           : Predicting tokens probabilities
#
# For instance, GPT-3 uses a form of tokenization called Byte Pair Encoding (BPE), which tokenizes text into subwords. 
# These subwords are sequences of characters that frequently appear together. This approach is a middle ground between 
# character-level and word-level tokenization, balancing efficiency and versatility.

import unidecode
import string
import torch
import json
import re
import os

class vocabulario:
    def __init__(self, character_level: bool = True):
        
        #initiate the index to token dict
        ## <PAD> -> padding, used for padding the shorter sentences in a batch to match the length of longest sentence in the batch
        ## <SOS> -> start token, added in front of each sentence to signify the start of sentence
        ## <EOS> -> End of sentence token, added to the end of each sentence to signify the end of sentence
        ## <UNK> -> words which are not found in the vocab are replace by this token
        ## <mask> -> the MASK token to add into the Vocabulary; indicates a position that will not be used in updating the model's parameters


        self.character_level = character_level

        self._token_to_idx = {}

        self._idx_to_token = {idx: token 
                              for token, idx in self._token_to_idx.items()}

        self._pad_token = "<pad>"
        self.pad_index = self.add_token(self._pad_token)
        
        self._start_sentence = "<sos>"
        self.start_index = self.add_token(self._start_sentence)

        self._end_sentence = "<eos>"
        self.end_index = self.add_token(self._end_sentence)

        self._unk_token = "<unk>"
        self.unk_index = self.add_token(self._unk_token)

        self._space_token = " "
        self.space_index = self.add_token(self._space_token)

        for i in range(10):
            self.add_token(str(i))
        for i in range(26):
            self.add_token(chr(ord('a') + i))
        for punc in string.punctuation:
            self.add_token(punc)

    def __str__(self) -> str:
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self) -> int:
        return len(self._token_to_idx)

    def add_token(self, token: str) -> int:
        """Update mapping dicts based on the token.

        Args:
            token (str): the item to add into the Vocabulary
        Returns:
            index (int): the integer corresponding to the token
        """
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index
    
    def tokenize(self,text: str) -> list[int]:

        text = text.lower()
        text = re.sub('&quot;', ' ', text)
        text = re.sub('&amp;','',text)
        text = re.sub('[||]','',text)
        text = text.replace('.', ' . ')
        text = text.split()
        text = ' '.join([unidecode.unidecode(word) for word in text])

        if self.character_level:
            return [self._token_to_idx[character] for character in text]
        else:
            return [self.add_token(word) for word in text.split()]
    

    def lookup_token(self, token: str) -> int:
        """Retrieve the index associated with the token 
          or the UNK index if token isn't present.
        
        Args:
            token (str): the token to look up 
        Returns:
            index (int): the index corresponding to the token
        Notes:
            `unk_index` needs to be >=0 (having been added into the Vocabulary) 
              for the UNK functionality 
        """
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

    def lookup_index(self, index: int) -> str:
        """Return the token associated with the index
        
        Args: 
            index (int): the index to look up
        Returns:
            token (str): the token corresponding to the index
        Raises:
            KeyError: if the index is not in the Vocabulary
        """
        if type(index) == torch.Tensor:
            index = index.item()
        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)

        
        return self._idx_to_token[index]
    
    def to_serializable(self, folder_path: str, file_name: str) -> None:
        """ save a dictionary serialized """
        file_path = os.path.join(folder_path, file_name)

        series = {'character_level' : 'word' if not self.character_level else 'character',
                'unk_token': self._unk_token, 
                'pad_token' : self._pad_token,
                'start_token': self._start_sentence,
                'end_token': self._end_sentence,
                'token_to_idx': self._token_to_idx}

        print(f'Saving Dictionary {file_path}', flush = True)
        
        # Salva o dicionário como um arquivo JSON
        with open(file_path, 'w') as json_file:
            json.dump(series, json_file, indent=4, ensure_ascii=False)
    
    def load_from_json(self, file_path: str) -> None:

        try:
            # Carrega os dados do arquivo JSON
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        # Atribui os valores ao objeto
        self._unk_token = data.get('unk_token')
        self._pad_token = data.get('pad_token')
        self._start_sentence = data.get('start_token')
        self._end_sentence = data.get('end_token')
        self._token_to_idx = data.get('token_to_idx', {})
        self.unk_index = self._token_to_idx[self._unk_token]
        self._idx_to_token = {idx: token 
                          for token, idx in self._token_to_idx.items()}
# %%
