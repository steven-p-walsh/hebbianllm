#!/usr/bin/env python3
"""Test tokenizer spacing behavior."""

import sys
from pathlib import Path

# Add hebbianllm to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from experiments.conversational_learning.utils.biologically_inspired_tokenizer import BiologicalTokenizer

def test_tokenizer_spacing():
    """Test if tokenizer preserves word spacing."""
    
    tokenizer = BiologicalTokenizer(max_vocab_size=1000)
    
    # Test phrases with spaces
    test_phrases = [
        "hello world",
        "the cat sat",
        "good morning",
        "how are you",
        "this is a test"
    ]
    
    print("Testing tokenizer spacing behavior:")
    print("=" * 50)
    
    for phrase in test_phrases:
        print(f"\nOriginal: '{phrase}'")
        
        # Encode
        encoded = tokenizer.encode(phrase)
        print(f"Encoded: {encoded}")
        
        # Show what tokens these IDs represent
        token_names = []
        for token_id in encoded:
            if token_id in tokenizer.id_to_pattern:
                token_names.append(tokenizer.id_to_pattern[token_id])
            else:
                token_names.append(f"<UNK:{token_id}>")
        print(f"Tokens: {token_names}")
        
        # Decode
        decoded = tokenizer.decode(encoded)
        print(f"Decoded: '{decoded}'")
        
        # Check if spaces are preserved
        if phrase == decoded:
            print("✅ Spacing preserved correctly")
        else:
            print("❌ Spacing issue detected!")
            print(f"   Expected: '{phrase}'")
            print(f"   Got:      '{decoded}'")
    
    print("\n" + "=" * 50)
    print("Tokenizer vocabulary size:", tokenizer.get_vocab_size())

if __name__ == "__main__":
    test_tokenizer_spacing()