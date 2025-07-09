"""
Biologically-Inspired Tokenizer

Implements character-level tokenization with biological constraints:
- Character-level processing (more biologically plausible)
- Temporal patterns for phonetics
- Hierarchical encoding (characters -> morphemes -> words)
- Support for special tokens and punctuation
"""

import string
import json
from typing import List, Dict, Tuple, Optional
import numpy as np


class BiologicalTokenizer:
    """
    Character-level tokenizer designed for biological neural networks.
    
    Features:
    - Character-level encoding (closer to biological speech processing)
    - Special tokens for sequence boundaries and unknown characters
    - Hierarchical vocabulary organization
    - Support for common punctuation and symbols
    """
    
    def __init__(self, max_vocab_size: int = 1000):
        self.max_vocab_size = max_vocab_size
        
        # Build character vocabulary
        self._build_vocabulary()
        
        # Token mappings
        self.char_to_id = {}
        self.id_to_char = {}
        self._create_mappings()
        
        # Sequence processing parameters
        self.max_sequence_length = 512
        
    def _build_vocabulary(self):
        """Build character-based vocabulary."""
        
        # Special tokens (must come first for consistent IDs)
        self.special_tokens = [
            '<PAD>',    # 0 - Padding token
            '<UNK>',    # 1 - Unknown character
            '<BOS>',    # 2 - Beginning of sequence
            '<EOS>',    # 3 - End of sequence
            '<SPACE>',  # 4 - Space character (explicit)
        ]
        
        # Basic character sets
        self.basic_chars = [
            # Lowercase letters (most common)
            *string.ascii_lowercase,
            # Uppercase letters
            *string.ascii_uppercase,
            # Digits
            *string.digits,
            # Common punctuation
            '.', ',', '!', '?', ';', ':', "'", '"',
            '-', '(', ')', '[', ']', '{', '}',
            # Additional useful characters
            '\n', '\t', '@', '#', '$', '%', '&', '*', '+', '=', '/', '\\', '|', '~', '`'
        ]
        
        # Combine vocabulary
        self.vocabulary = self.special_tokens + self.basic_chars
        
        # Trim to max size if needed
        if len(self.vocabulary) > self.max_vocab_size:
            self.vocabulary = self.vocabulary[:self.max_vocab_size]
        
        print(f"Vocabulary size: {len(self.vocabulary)}")
        print(f"Special tokens: {len(self.special_tokens)}")
        print(f"Character tokens: {len(self.basic_chars)}")
    
    def _create_mappings(self):
        """Create bidirectional mappings between characters and IDs."""
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocabulary)}
        self.id_to_char = {idx: char for idx, char in enumerate(self.vocabulary)}
        
        # Quick access to special token IDs
        self.pad_id = self.char_to_id['<PAD>']
        self.unk_id = self.char_to_id['<UNK>']
        self.bos_id = self.char_to_id['<BOS>']
        self.eos_id = self.char_to_id['<EOS>']
        self.space_id = self.char_to_id['<SPACE>']
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to list of token IDs.
        
        Args:
            text: Input text string
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of token IDs
        """
        if not text:
            return [self.bos_id, self.eos_id] if add_special_tokens else []
        
        # Convert to character IDs
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.bos_id)
        
        for char in text:
            if char == ' ':
                # Explicit space token
                token_ids.append(self.space_id)
            elif char in self.char_to_id:
                token_ids.append(self.char_to_id[char])
            else:
                # Unknown character
                token_ids.append(self.unk_id)
        
        if add_special_tokens:
            token_ids.append(self.eos_id)
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode list of token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text string
        """
        if not token_ids:
            return ""
        
        chars = []
        
        for token_id in token_ids:
            if token_id in self.id_to_char:
                char = self.id_to_char[token_id]
                
                # Handle special tokens
                if skip_special_tokens and char in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']:
                    continue
                elif char == '<SPACE>':
                    chars.append(' ')
                else:
                    chars.append(char)
            else:
                # Invalid token ID
                if not skip_special_tokens:
                    chars.append('<UNK>')
        
        return ''.join(chars)
    
    def encode_batch(self, texts: List[str], 
                    max_length: Optional[int] = None,
                    padding: bool = True,
                    truncation: bool = True) -> List[List[int]]:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            padding: Whether to pad sequences to same length
            truncation: Whether to truncate long sequences
            
        Returns:
            List of token ID sequences
        """
        if max_length is None:
            max_length = self.max_sequence_length
        
        encoded_batch = []
        
        for text in texts:
            # Encode individual text
            token_ids = self.encode(text)
            
            # Truncate if needed
            if truncation and len(token_ids) > max_length:
                token_ids = token_ids[:max_length-1] + [self.eos_id]
            
            encoded_batch.append(token_ids)
        
        # Pad to same length if requested
        if padding:
            max_len = max(len(seq) for seq in encoded_batch) if encoded_batch else 0
            max_len = min(max_len, max_length)
            
            for i, seq in enumerate(encoded_batch):
                if len(seq) < max_len:
                    # Pad with PAD tokens
                    encoded_batch[i] = seq + [self.pad_id] * (max_len - len(seq))
        
        return encoded_batch
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocabulary)
    
    def get_special_token_ids(self) -> Dict[str, int]:
        """Get mapping of special token names to IDs."""
        return {
            'pad': self.pad_id,
            'unk': self.unk_id,
            'bos': self.bos_id,
            'eos': self.eos_id,
            'space': self.space_id
        }
    
    def is_special_token(self, token_id: int) -> bool:
        """Check if token ID corresponds to a special token."""
        if token_id in self.id_to_char:
            char = self.id_to_char[token_id]
            return char in self.special_tokens
        return False
    
    def get_character_categories(self) -> Dict[str, List[int]]:
        """Get token IDs grouped by character categories."""
        categories = {
            'letters': [],
            'digits': [],
            'punctuation': [],
            'special': [],
            'space': []
        }
        
        for token_id, char in self.id_to_char.items():
            if char in self.special_tokens:
                if char == '<SPACE>':
                    categories['space'].append(token_id)
                else:
                    categories['special'].append(token_id)
            elif char.isalpha():
                categories['letters'].append(token_id)
            elif char.isdigit():
                categories['digits'].append(token_id)
            else:
                categories['punctuation'].append(token_id)
        
        return categories
    
    def analyze_text(self, text: str) -> Dict:
        """Analyze text and return statistics."""
        token_ids = self.encode(text, add_special_tokens=False)
        
        # Character category counts
        categories = self.get_character_categories()
        category_counts = {}
        
        for category, cat_ids in categories.items():
            category_counts[category] = sum(1 for tid in token_ids if tid in cat_ids)
        
        # Basic statistics
        stats = {
            'text_length': len(text),
            'token_count': len(token_ids),
            'unique_tokens': len(set(token_ids)),
            'category_counts': category_counts,
            'compression_ratio': len(token_ids) / len(text) if text else 0
        }
        
        return stats
    
    def save_vocabulary(self, filepath: str):
        """Save vocabulary to JSON file."""
        vocab_data = {
            'vocabulary': self.vocabulary,
            'char_to_id': self.char_to_id,
            'id_to_char': {str(k): v for k, v in self.id_to_char.items()},  # JSON needs string keys
            'special_tokens': self.special_tokens,
            'max_vocab_size': self.max_vocab_size
        }
        
        with open(filepath, 'w') as f:
            json.dump(vocab_data, f, indent=2)
        
        print(f"Vocabulary saved to {filepath}")
    
    def load_vocabulary(self, filepath: str):
        """Load vocabulary from JSON file."""
        with open(filepath, 'r') as f:
            vocab_data = json.load(f)
        
        self.vocabulary = vocab_data['vocabulary']
        self.char_to_id = vocab_data['char_to_id']
        self.id_to_char = {int(k): v for k, v in vocab_data['id_to_char'].items()}  # Convert back to int keys
        self.special_tokens = vocab_data['special_tokens']
        self.max_vocab_size = vocab_data['max_vocab_size']
        
        # Recreate special token shortcuts
        self.pad_id = self.char_to_id['<PAD>']
        self.unk_id = self.char_to_id['<UNK>']
        self.bos_id = self.char_to_id['<BOS>']
        self.eos_id = self.char_to_id['<EOS>']
        self.space_id = self.char_to_id['<SPACE>']
        
        print(f"Vocabulary loaded from {filepath}")


class SequenceProcessor:
    """
    Utilities for processing sequences for the Hebbian network.
    
    Handles:
    - Sequence batching
    - Temporal encoding
    - Context windowing
    - Sequence augmentation
    """
    
    def __init__(self, tokenizer: BiologicalTokenizer):
        self.tokenizer = tokenizer
        self.max_context_length = 128
    
    def create_training_pairs(self, token_ids: List[int], 
                            context_length: int = 10) -> List[Tuple[List[int], int]]:
        """
        Create input-target pairs for training.
        
        Args:
            token_ids: Full sequence of token IDs
            context_length: Length of context window
            
        Returns:
            List of (context, target) pairs
        """
        pairs = []
        
        for i in range(context_length, len(token_ids)):
            context = token_ids[i-context_length:i]
            target = token_ids[i]
            pairs.append((context, target))
        
        return pairs
    
    def augment_sequence(self, token_ids: List[int], 
                        noise_rate: float = 0.1) -> List[int]:
        """
        Apply data augmentation to a sequence.
        
        Args:
            token_ids: Original sequence
            noise_rate: Probability of applying noise to each token
            
        Returns:
            Augmented sequence
        """
        augmented = token_ids.copy()
        
        for i in range(len(augmented)):
            if np.random.random() < noise_rate:
                # Apply random augmentation
                aug_type = np.random.choice(['substitute', 'delete', 'insert'])
                
                if aug_type == 'substitute' and not self.tokenizer.is_special_token(augmented[i]):
                    # Substitute with random character from same category
                    categories = self.tokenizer.get_character_categories()
                    for category, cat_ids in categories.items():
                        if augmented[i] in cat_ids and len(cat_ids) > 1:
                            # Replace with random token from same category
                            replacement_options = [tid for tid in cat_ids if tid != augmented[i]]
                            augmented[i] = np.random.choice(replacement_options)
                            break
                
                elif aug_type == 'delete' and len(augmented) > 1:
                    # Delete token (don't delete if it would make sequence empty)
                    if not self.tokenizer.is_special_token(augmented[i]):
                        augmented.pop(i)
                        break
                
                elif aug_type == 'insert':
                    # Insert random character
                    categories = self.tokenizer.get_character_categories()
                    all_non_special = []
                    for category in ['letters', 'digits', 'punctuation']:
                        all_non_special.extend(categories.get(category, []))
                    
                    if all_non_special:
                        random_token = np.random.choice(all_non_special)
                        augmented.insert(i, random_token)
                        break
        
        return augmented
    
    def create_conversation_context(self, conversation_history: List[str], 
                                  max_history: int = 5) -> List[int]:
        """
        Create context from conversation history.
        
        Args:
            conversation_history: List of previous utterances
            max_history: Maximum number of previous utterances to include
            
        Returns:
            Encoded context sequence
        """
        # Take recent history
        recent_history = conversation_history[-max_history:] if conversation_history else []
        
        # Combine into single context
        context_text = " ".join(recent_history)
        
        # Encode and truncate if needed
        context_ids = self.tokenizer.encode(context_text, add_special_tokens=False)
        
        if len(context_ids) > self.max_context_length:
            context_ids = context_ids[-self.max_context_length:]
        
        return context_ids
    
    def prepare_batch_for_network(self, sequences: List[List[int]], 
                                 target_length: int = 64) -> np.ndarray:
        """
        Prepare batch of sequences for network processing.
        
        Args:
            sequences: List of token ID sequences
            target_length: Target sequence length
            
        Returns:
            Padded and batched sequences as numpy array
        """
        # Pad or truncate to target length
        processed_sequences = []
        
        for seq in sequences:
            if len(seq) > target_length:
                # Truncate
                processed_seq = seq[:target_length]
            else:
                # Pad
                padding_length = target_length - len(seq)
                processed_seq = seq + [self.tokenizer.pad_id] * padding_length
            
            processed_sequences.append(processed_seq)
        
        return np.array(processed_sequences)