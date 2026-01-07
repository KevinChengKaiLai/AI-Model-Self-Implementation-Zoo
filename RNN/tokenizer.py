# tokenizer.py
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle


class CharTokenizer:
    def __init__(self):
        
        self.special_tokens = ['<PAD>', '<UNK>']
        import string
        chars = string.ascii_lowercase + string.ascii_uppercase + string.digits + string.punctuation + ' '
        self.vocab = self.special_tokens + list(chars)

        self.char2idx = {char:idx for idx,char in enumerate(self.vocab)}
        self.idx2char = {idx:char for idx,char in enumerate(self.vocab)}

        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'

        self.pad_idx = self.char2idx[self.pad_token]
        self.unk_idx = self.char2idx[self.unk_token]

    def encode(self, text):
        return [self.char2idx.get(char,self.unk_idx) for char in text]

    def decode(self, indices):
        return "".join([self.idx2char[idx] for idx in indices])
    
    def __len__(self):
        return len(self.vocab)
    

class BPETokenizer:
    # represent a word As a tuple of characters with end marker during bpe training
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.merges = {}  # Store merge operations: {('l', 'o'): 'lo'}
        self.vocab = {}   # Final vocabulary: {'hello': 0, 'world': 1, ...}
    
    def _get_word_frequencies(self,texts):
        """
        Count word frequencies from texts
        
        Args:
            texts: List[str] - ["hello world", "hello"]
        
        Returns:
            Dict[tuple, int] - {('h','e','l','l','o','</w>'): 2, ...}
        """
        from collections import defaultdict
        word_freqs = defaultdict(int)
        
        for text in texts:
            words = text.split()  # Split on whitespace
            for word in words:
                # TODO: Convert word to tuple of characters, Add end-of-word marker '</w>'
                word_tuple = tuple(word) + ('</w>',)

                # TODO: Count frequency
                word_freqs[word_tuple] += 1
        
        return dict(word_freqs)
    
    def _get_pair_frequencies(self, word_freqs:dict):
        """
        Count frequencies of adjacent character pairs
        
        Args:
            word_freqs: Dict[tuple, int] - {('h','e','l','l','o','</w>'): 2}
        
        Returns:
            Dict[tuple, int] - {('h','e'): 2, ('e','l'): 2, ('l','l'): 2, ...}
        """
        from collections import defaultdict
        pair_freqs = defaultdict(int)

        for word, freq in word_freqs.items():
            n = len(word)
            for i in range(n-1):
                temp = word[i:i+2]
                pair_freqs[temp] += freq
        
        # return a regular dict in case bug happen
        return dict(pair_freqs)    
    
    def _merge_pair(self, word_freqs:dict, pair:tuple):
        """
        Merge a specific pair in all words
        
        Args:
            word_freqs: Dict[tuple, int] - {('h','e','l','l','o','</w>'): 2}
            pair: tuple - ('l', 'l') - the pair to merge
        
        Returns:
            Dict[tuple, int] - Updated word_freqs with pair merged
        
        Example:
            Input:  {('h','e','l','l','o','</w>'): 2}, pair=('l','l')
            Output: {('h','e','ll','o','</w>'): 2}
        """
        new_word_freqs = {}

        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i],word[i+1]) == pair:
                    new_word.append(word[i]+word[i+1])
                    i+=1
                else:
                    new_word.append(word[i])

                i += 1

            new_word_freqs[tuple(new_word)] = new_word_freqs.get(tuple(new_word), 0) + freq
        
        return new_word_freqs
        
            
    def train(self, texts):
        """
        Learn BPE merges from a corpus
        
        Args:
            texts: List[str] - e.g., ["hello world", "hello there"]
        """
        # Step 1: Initialize vocabulary with characters
        word_freqs = self._get_word_frequencies(texts)
        # Step 2: Count word frequencies
        import string
        vocab = set()
        vocab.update(string.ascii_lowercase)
        vocab.update(string.ascii_uppercase)
        vocab.update(string.digits)
        vocab.update(string.punctuation)
        vocab.update(string.punctuation)



        for word in word_freqs.keys():
            vocab.update(word)
        print(f"Initial vocab size: {len(vocab)}")
        print(f"Target vocab size: {self.vocab_size}")
    
        # Step 3: Iteratively merge most frequent pairs
        num_merges = self.vocab_size - len(vocab)
        print(f'we will perform {num_merges} merges \n')

        for i in range(num_merges):
            pair_freqs = self._get_pair_frequencies(word_freqs)

            if not pair_freqs:
                print('Nothing to pair anymore!')
                break

            # get highest freq pair --- this is pair only, no freq
            freqest_pair = max(pair_freqs, key=lambda x: pair_freqs[x])

            # merge this pair and update the word_freqs
            word_freqs = self._merge_pair(word_freqs, freqest_pair)
            # update self merge

            self.merges[freqest_pair] = "".join(freqest_pair)
            # print(f"Merge {i+1}: {freqest_pair} → {''.join(freqest_pair)}")
            # print(f"Word freqs after merge {i+1}: {word_freqs}")

    

        # Step 4: Build final vocabulary
        for word in word_freqs.keys():
            vocab.update(word)
        for pair, merged_pair in self.merges.items():
            vocab.add(merged_pair)
        
        self.vocab = {token:idx for idx,token in enumerate(sorted(vocab))}
        self.idx =  {idx:token for idx,token in enumerate(sorted(vocab))}
    
    def encode(self, text: str):
        """
        Tokenize text using learned merges
        
        Args:
            text: str
        Returns:
            List[int] - token IDs
        """
        token_list = []
        word_list = text.split()
        for word in word_list:
            char_list = [c for c in word] + ['</w>']
            # ['l', 'o', 'w', 'e', 's', 't', '</w>']
            
            for pair, merged_token in self.merges.items():
                merged_char_list = []
                i = 0
                while i < len(char_list)-1:
                    c1,c2 = char_list[i], char_list[i+1]
                    if (c1,c2) == pair:
                        merged_char_list.append(merged_token)
                        i += 2
                    else: # add the char in i position and go next
                        merged_char_list.append(c1)
                        i += 1
                
                merged_char_list += char_list[i:]
                char_list = merged_char_list
                
            token_list.extend(char_list)
        
        tokenized_list = [self.vocab[i] for i in token_list]

        return tokenized_list
        
    
    def decode(self, token_ids):
        """
        Convert token IDs back to text
        
        Args:
            token_ids: List[int]
        Returns:
            str
        """
        tokens = [self.idx[id] for id in token_ids]
        text = "".join(tokens)
        
        return text.replace("</w>"," ").strip()
    
    def save(self, filepath):
        data = {'vocab': self.vocab, 'merges': self.merges}

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.vocab = data['vocab']
            self.merges = data['merges']
            self.idx = {v:k for k,v in self.vocab.items()} # rebuild idx from vocab


if __name__ == '__main__':
    texts = ["low low low low low lower lowest"]
    tokenizer = BPETokenizer(vocab_size=200)
    tokenizer.train(texts)
    
    print("\nLearned merges:")
    for pair, merged in tokenizer.merges.items():
        print(f"  {pair} → {merged}")
    
    print(f"\nFinal vocab size: {len(tokenizer.vocab)}")
    print(f' this is final vocab: \n {tokenizer.vocab}')

    original = "low low lower lowest"
    encoded = tokenizer.encode(original)
    decoded = tokenizer.decode(encoded)
    print(f"Original: '{original}'")
    print(f"Decoded:  '{decoded}'")
    print(f"Match: {original == decoded}")

    encoded = tokenizer.encode("LOWESTLOWEST LOWEST")  # Uppercase
    decoded = tokenizer.decode(encoded)
    print(encoded)
    print(decoded)

    test_texts = [
    "Hello, world!",
    "The quick brown fox jumps.",
    "COVID-19 is a pandemic.",
    "I love Python 3.11!",
]

    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"Original: {text}")
        print(f"Decoded:  {decoded}")
        print(f"Match: {text == decoded}")
        print(f"Tokens: {len(encoded)}")
        print()

    tokenizer.save('bpe_tokenizer.pkl')

    # Load into a new tokenizer
    new_tokenizer = BPETokenizer(vocab_size=500)
    new_tokenizer.load('bpe_tokenizer.pkl')

    # Test
    print(new_tokenizer.encode("Hello world"))
    print(new_tokenizer.decode(new_tokenizer.encode("Hello world")))