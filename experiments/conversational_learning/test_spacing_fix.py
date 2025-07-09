#!/usr/bin/env python3
"""Test spacing fix for neural network generation."""

import sys
from pathlib import Path

# Add hebbianllm to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from experiments.conversational_learning.utils.biologically_inspired_tokenizer import BiologicalTokenizer

def test_decode_with_special_tokens():
    """Test decode with skip_special_tokens=False."""
    
    tokenizer = BiologicalTokenizer(max_vocab_size=1000)
    
    # Test phrase with spaces
    test_phrase = "hello world"
    
    print("Testing decode with skip_special_tokens=False:")
    print("=" * 50)
    
    print(f"Original: '{test_phrase}'")
    
    # Encode
    encoded = tokenizer.encode(test_phrase)
    print(f"Encoded: {encoded}")
    
    # Show what tokens these IDs represent
    token_names = []
    for token_id in encoded:
        if token_id in tokenizer.id_to_pattern:
            token_names.append(tokenizer.id_to_pattern[token_id])
        else:
            token_names.append(f"<UNK:{token_id}>")
    print(f"Tokens: {token_names}")
    
    # Decode with skip_special_tokens=True (old behavior)
    decoded_skip = tokenizer.decode(encoded, skip_special_tokens=True)
    print(f"Decoded (skip=True):  '{decoded_skip}'")
    
    # Decode with skip_special_tokens=False (new behavior)
    decoded_no_skip = tokenizer.decode(encoded, skip_special_tokens=False)
    print(f"Decoded (skip=False): '{decoded_no_skip}'")
    
    # Clean up like the plastic learner does
    cleaned = decoded_no_skip.replace('<BOS>', '').replace('<EOS>', '').replace('<UNK>', '').strip()
    print(f"Cleaned version:      '{cleaned}'")
    
    # Check results
    if decoded_skip == test_phrase:
        print("✅ Skip=True preserves spacing")
    else:
        print("❌ Skip=True loses spacing")
    
    if cleaned == test_phrase:
        print("✅ Skip=False + cleanup preserves spacing")
    else:
        print("❌ Skip=False + cleanup loses spacing")

if __name__ == "__main__":
    test_decode_with_special_tokens()