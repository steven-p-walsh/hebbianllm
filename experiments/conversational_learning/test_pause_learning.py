#!/usr/bin/env python3
"""Test PAUSE token learning in BiologicalTokenizer."""

import sys
from pathlib import Path

# Add hebbianllm to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from experiments.conversational_learning.utils.biologically_inspired_tokenizer import BiologicalTokenizer

def test_pause_learning():
    """Test if tokenizer learns patterns with PAUSE tokens."""
    
    tokenizer = BiologicalTokenizer(max_vocab_size=1000)
    
    # Test phrases that should create PAUSE patterns
    test_phrases = [
        "hello world",
        "hello world",  # Repeat to increase frequency
        "hello world",
        "the cat",
        "the cat",
        "the cat",
        "good morning",
        "good morning"
    ]
    
    print("Testing PAUSE token learning:")
    print("=" * 50)
    
    print("Initial vocab size:", tokenizer.get_vocab_size())
    
    # Process phrases multiple times to trigger learning
    for i, phrase in enumerate(test_phrases):
        print(f"\n[{i+1}] Processing: '{phrase}'")
        
        # Encode (this triggers learning)
        encoded = tokenizer.encode(phrase)
        
        # Show tokens
        token_names = []
        for token_id in encoded:
            if token_id in tokenizer.id_to_pattern:
                token_names.append(tokenizer.id_to_pattern[token_id])
            else:
                token_names.append(f"<UNK:{token_id}>")
        
        print(f"   Tokens: {token_names}")
        print(f"   Vocab size: {tokenizer.get_vocab_size()}")
    
    print("\n" + "=" * 50)
    print("Pattern learning results:")
    
    # Look for patterns that include PAUSE
    pause_patterns = []
    for pattern_id, pattern in tokenizer.id_to_pattern.items():
        if '<PAUSE>' in pattern:
            pause_patterns.append(pattern)
    
    if pause_patterns:
        print("✅ Found PAUSE patterns:")
        for pattern in pause_patterns:
            print(f"   '{pattern}' (id: {tokenizer.pattern_to_id[pattern]})")
    else:
        print("❌ No PAUSE patterns learned")
    
    # Check pattern combinations
    print("\nPattern combinations involving PAUSE:")
    for first_pattern, second_patterns in list(tokenizer.pattern_combinations.items())[:10]:
        for second_pattern, count in second_patterns.items():
            if '<PAUSE>' in first_pattern or '<PAUSE>' in second_pattern:
                print(f"   '{first_pattern}' + '{second_pattern}' = {count} times")
    
    # Test if learned patterns are used in generation
    print("\n" + "=" * 50)
    print("Testing generation with learned patterns:")
    
    test_decode = "hello world again"
    encoded = tokenizer.encode(test_decode)
    decoded = tokenizer.decode(encoded, skip_special_tokens=False)
    cleaned = decoded.replace('<BOS>', '').replace('<EOS>', '').replace('<UNK>', '').strip()
    
    print(f"Original: '{test_decode}'")
    print(f"Encoded:  {encoded}")
    print(f"Decoded:  '{decoded}'")
    print(f"Cleaned:  '{cleaned}'")
    
    if cleaned == test_decode:
        print("✅ Spacing preserved correctly")
    else:
        print("❌ Spacing issue detected")

if __name__ == "__main__":
    test_pause_learning()