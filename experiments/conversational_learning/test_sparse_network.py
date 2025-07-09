#!/usr/bin/env python3
"""
Test Sparse Network Generation

Quick test to see what's happening with the sparse coding.
"""

import sys
import os
from pathlib import Path

# Add hebbianllm to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.plastic_snn import PlasticHebSNN
from utils.biologically_inspired_tokenizer import BiologicalTokenizer

def test_sparse_generation():
    print("🧪 Testing Sparse Network Generation")
    print("=" * 40)
    
    # Force GPU isolation
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    # Create small network
    tokenizer = BiologicalTokenizer(max_vocab_size=100)
    network = PlasticHebSNN(
        n_neurons=500,    # Smaller for testing
        vocab_size=100,   # Smaller vocab
        initial_connectivity=0.02
    )
    
    print(f"Network initialized: {network.n_neurons} neurons")
    
    # Test babbling generation
    print("\n1. Testing babbling mode...")
    try:
        babble_tokens = network._generate_babbling(max_length=3)
        print(f"Babble tokens: {babble_tokens}")
        
        if babble_tokens:
            babble_text = tokenizer.decode(babble_tokens, skip_special_tokens=True)
            print(f"Babble text: '{babble_text}'")
        else:
            print("No babble tokens generated")
            
    except Exception as e:
        print(f"Babbling failed: {e}")
    
    # Test simple input processing
    print("\n2. Testing input processing...")
    try:
        test_input = "hi"
        input_tokens = tokenizer.encode(test_input)
        print(f"Input tokens for '{test_input}': {input_tokens}")
        
        if input_tokens:
            result = network.process_tokens(input_tokens, learning=True)
            print(f"Processing result keys: {result.keys()}")
            print(f"Activity shape: {result['activity'].shape}")
            print(f"Active neurons: {(result['activity'] > 0).sum()}")
            
            # Test generation
            generated = network.generate_tokens(input_tokens[:1], max_length=2)
            print(f"Generated tokens: {generated}")
            
            if len(generated) > len(input_tokens):
                response_tokens = generated[len(input_tokens):]
                response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
                print(f"Response: '{response_text}'")
            else:
                print("No new tokens generated")
        
    except Exception as e:
        print(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sparse_generation()