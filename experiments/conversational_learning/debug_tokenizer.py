#!/usr/bin/env python3
"""
Debug Tokenizer Structure
"""

import sys
from pathlib import Path

# Add hebbianllm to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.biologically_inspired_tokenizer import BiologicalTokenizer

def debug_tokenizer():
    print("🔍 Debugging Tokenizer Structure")
    print("=" * 40)
    
    tokenizer = BiologicalTokenizer(max_vocab_size=100)
    
    print(f"Token mappings type: {type(tokenizer.token_mappings)}")
    print(f"Number of mappings: {len(tokenizer.token_mappings)}")
    
    # Show first 10 mappings
    print("\nFirst 10 token mappings:")
    for i, (pattern, token_id) in enumerate(tokenizer.token_mappings.items()):
        if i >= 10:
            break
        print(f"  '{pattern}' -> {token_id} (type: {type(token_id)})")
    
    # Test encoding/decoding
    print("\nTesting encode/decode:")
    test_words = ["hi", "ma", "da", "o", "a", "e"]
    for word in test_words:
        tokens = tokenizer.encode(word)
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"  '{word}' -> {tokens} -> '{decoded}'")
    
    # Show available patterns
    print("\nAvailable short patterns:")
    short_patterns = []
    for pattern, token_id in tokenizer.token_mappings.items():
        if isinstance(pattern, str) and len(pattern) <= 2 and pattern.isalpha():
            short_patterns.append((pattern, token_id))
            if len(short_patterns) >= 10:
                break
    
    for pattern, token_id in short_patterns:
        print(f"  '{pattern}' -> {token_id}")

if __name__ == "__main__":
    debug_tokenizer()