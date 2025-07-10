"""
Biologically-Inspired Tokenization

Instead of characters or pre-defined tokens, this implements a more biological approach:
- Syllable-like chunks that emerge from acoustic patterns
- Morpheme-aware segmentation
- Learned vocabulary that grows through experience
- Hierarchical organization: sounds -> syllables -> morphemes -> words
"""

import re
import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass


@dataclass
class SyllablePattern:
    """Represents a syllable-like sound pattern."""
    pattern: str
    frequency: int
    acoustic_features: List[float]  # Simplified acoustic representation
    

class BiologicalTokenizer:
    """
    Tokenizer that learns syllable-like patterns from experience.
    
    Biological inspiration:
    - Starts with basic phoneme-like units
    - Learns common patterns through exposure
    - Builds hierarchical representations
    - Adapts vocabulary through usage
    """
    
    def __init__(self, max_vocab_size: int = 2000):
        self.max_vocab_size = max_vocab_size
        
        # Start with basic phoneme-like patterns
        self.base_patterns = self._init_base_patterns()
        
        # Learned vocabulary
        self.learned_patterns = {}  # pattern -> SyllablePattern
        self.pattern_to_id = {}
        self.id_to_pattern = {}
        
        # Special tokens
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1, 
            '<BOS>': 2,
            '<EOS>': 3,
            '<PAUSE>': 4,  # For natural speech pauses
        }
        
        # Usage statistics for adaptation
        self.pattern_usage = Counter()
        self.pattern_combinations = defaultdict(Counter)
        
        # Initialize vocabulary
        self._build_initial_vocabulary()
        
        print(f"Biological tokenizer initialized with {len(self.pattern_to_id)} base patterns")
    
    def _init_base_patterns(self) -> List[str]:
        """Initialize with basic phoneme-like patterns."""
        # Common English syllable patterns and phonemes
        vowels = ['a', 'e', 'i', 'o', 'u', 'y']
        consonants = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 
                     'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'z']
        
        patterns = []
        
        # Single phonemes
        patterns.extend(vowels)
        patterns.extend(consonants)
        
        # Common syllable patterns (CV, CVC, etc.)
        for c in consonants[:10]:  # Most common consonants
            for v in vowels:
                patterns.append(c + v)  # CV
                for c2 in ['t', 'n', 's', 'r', 'l']:  # Common ending consonants
                    patterns.append(c + v + c2)  # CVC
        
        # Common morphemes and function words
        function_words = [
            'the', 'and', 'is', 'in', 'to', 'of', 'for', 'with', 'on', 'at',
            'it', 'you', 'that', 'this', 'be', 'are', 'was', 'have', 'had',
            'ing', 'ed', 'er', 'ly', 'un', 're', 'pre', 'dis'
        ]
        patterns.extend(function_words)
        
        return patterns
    
    def _build_initial_vocabulary(self):
        """Build initial vocabulary from base patterns."""
        # Add special tokens first
        current_id = 0
        for token, token_id in self.special_tokens.items():
            self.pattern_to_id[token] = token_id
            self.id_to_pattern[token_id] = token
            current_id = max(current_id, token_id + 1)
        
        # Add base patterns
        for pattern in self.base_patterns:
            if pattern not in self.pattern_to_id:
                self.pattern_to_id[pattern] = current_id
                self.id_to_pattern[current_id] = pattern
                current_id += 1
    
    def _segment_text_biologically(self, text: str) -> List[str]:
        """
        Segment text into syllable-like chunks using biological principles.
        
        This mimics how humans naturally chunk speech:
        - Look for vowel centers (syllable nuclei)
        - Respect morpheme boundaries
        - Consider rhythm and stress patterns
        """
        if not text:
            return []
        
        text = text.lower().strip()
        segments = []
        
        # First pass: identify known patterns (longest match first)
        remaining = text
        i = 0
        
        while i < len(text):
            # Try to match known patterns (longest first)
            best_match = None
            best_length = 0
            
            for pattern in sorted(self.pattern_to_id.keys(), key=len, reverse=True):
                if pattern in self.special_tokens:
                    continue
                if text[i:].startswith(pattern) and len(pattern) > best_length:
                    best_match = pattern
                    best_length = len(pattern)
            
            if best_match:
                segments.append(best_match)
                i += best_length
            else:
                # Fall back to character level for unknown parts
                if text[i].isalpha():
                    segments.append(text[i])
                elif text[i].isspace():
                    segments.append('<PAUSE>')
                i += 1
        
        # Second pass: learn new patterns from combinations
        self._learn_from_segments(segments)
        
        return segments
    
    def _learn_from_segments(self, segments: List[str]):
        """Learn new patterns from observed segment combinations."""
        # Look for frequently occurring bigrams and trigrams
        for i in range(len(segments) - 1):
            # Learn patterns with and without PAUSE tokens
            # This allows the network to learn proper spacing patterns
            bigram = segments[i] + segments[i + 1]
            self.pattern_combinations[segments[i]][segments[i + 1]] += 1
            
            # For patterns without PAUSE, require more frequency
            # PAUSE patterns are prioritized with lower frequency requirements
            min_frequency = 3 if '<PAUSE>' not in bigram else 1
            
            # If this combination is frequent enough, consider it a new pattern
            if (len(bigram) <= 8 and  # Reasonable length (allow longer for PAUSE patterns)
                self.pattern_combinations[segments[i]][segments[i + 1]] >= min_frequency and  # Seen multiple times
                bigram not in self.pattern_to_id and  # Not already in vocab
                len(self.pattern_to_id) < self.max_vocab_size):  # Room in vocab
                
                self._add_new_pattern(bigram)
        
        # Also look for trigrams (including those with PAUSE tokens)
        for i in range(len(segments) - 2):
            trigram = segments[i] + segments[i + 1] + segments[i + 2]
            
            # Track trigram frequency
            if trigram not in self.pattern_combinations:
                self.pattern_combinations[trigram] = Counter()
            self.pattern_combinations[trigram]['_trigram_count'] += 1
            
            # For patterns with PAUSE, require lower frequency
            # PAUSE patterns are learned more aggressively
            min_frequency = 2 if '<PAUSE>' not in trigram else 1
            
            # Learn trigram if frequent enough
            if (len(trigram) <= 12 and  # Reasonable length (allow longer for PAUSE patterns)
                self.pattern_combinations[trigram]['_trigram_count'] >= min_frequency and  # Seen multiple times
                trigram not in self.pattern_to_id and  # Not already in vocab
                len(self.pattern_to_id) < self.max_vocab_size):  # Room in vocab
                
                self._add_new_pattern(trigram)
    
    def _add_new_pattern(self, pattern: str):
        """Add a new learned pattern to the vocabulary with priority for pause patterns."""
        new_id = len(self.pattern_to_id)
        self.pattern_to_id[pattern] = new_id
        self.id_to_pattern[new_id] = pattern
        
        # Give extra boost to patterns containing pause tokens
        if '<PAUSE>' in pattern:
            # Immediately increase usage to prioritize pause patterns
            self.pattern_usage[pattern] += 5  # Boost usage count
            print(f"Learned new PAUSE pattern: '{pattern}' (id: {new_id}) - PRIORITIZED")
        else:
            print(f"Learned new pattern: '{pattern}' (id: {new_id})")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs using biological segmentation."""
        if not text:
            return [self.special_tokens['<BOS>'], self.special_tokens['<EOS>']] if add_special_tokens else []
        
        # Segment text biologically
        segments = self._segment_text_biologically(text)
        
        # Convert to IDs
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.special_tokens['<BOS>'])
        
        for segment in segments:
            if segment in self.pattern_to_id:
                token_id = self.pattern_to_id[segment]
                self.pattern_usage[segment] += 1  # Track usage
            else:
                token_id = self.special_tokens['<UNK>']
            
            token_ids.append(token_id)
        
        if add_special_tokens:
            token_ids.append(self.special_tokens['<EOS>'])
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        if not token_ids:
            return ""
        
        segments = []
        
        for token_id in token_ids:
            if token_id in self.id_to_pattern:
                pattern = self.id_to_pattern[token_id]
                
                if pattern == '<PAUSE>':
                    segments.append(' ')
                elif skip_special_tokens and pattern in self.special_tokens:
                    continue
                else:
                    segments.append(pattern)
            else:
                if not skip_special_tokens:
                    segments.append('<UNK>')
        
        return ''.join(segments)
    
    def get_vocab_size(self) -> int:
        """Get current vocabulary size."""
        return len(self.pattern_to_id)
    
    def adapt_vocabulary(self, min_usage: int = 2):
        """
        Adapt vocabulary based on usage patterns.
        Remove rarely used patterns, promote frequent combinations.
        """
        # Find rarely used patterns
        patterns_to_remove = []
        for pattern, usage in self.pattern_usage.items():
            if (usage < min_usage and 
                pattern not in self.special_tokens and 
                '<PAUSE>' not in pattern):  # Protect pause patterns from removal
                patterns_to_remove.append(pattern)
        
        # Remove rarely used patterns (keeping space for new ones)
        for pattern in patterns_to_remove[:10]:  # Don't remove too many at once
            if pattern in self.pattern_to_id:
                old_id = self.pattern_to_id[pattern]
                del self.pattern_to_id[pattern]
                del self.id_to_pattern[old_id]
                print(f"Removed rarely used pattern: '{pattern}'")
        
        # Look for new patterns to promote from combinations
        candidate_patterns = []
        for first_pattern, second_patterns in self.pattern_combinations.items():
            # Skip internal counters but allow PAUSE patterns
            if '_trigram_count' in second_patterns:
                continue
                
            for second_pattern, count in second_patterns.items():
                if second_pattern == '_trigram_count':
                    continue
                    
                # Allow PAUSE patterns but require lower frequency
                # Prioritize pause patterns with even lower requirements
                min_count = 3 if '<PAUSE>' not in (first_pattern + second_pattern) else 1
                
                if count >= min_count:  # Frequently seen together
                    new_pattern = first_pattern + second_pattern
                    if (len(new_pattern) <= 12 and  # Allow longer for PAUSE patterns
                        new_pattern not in self.pattern_to_id and
                        new_pattern not in [p for p, _ in candidate_patterns]):
                        candidate_patterns.append((new_pattern, count))
        
        # Add most frequent new patterns
        candidate_patterns.sort(key=lambda x: x[1], reverse=True)
        for pattern, count in candidate_patterns[:5]:  # Add top 5
            if len(self.pattern_to_id) < self.max_vocab_size:
                self._add_new_pattern(pattern)
    
    def get_pattern_stats(self) -> Dict:
        """Get statistics about learned patterns."""
        return {
            'vocab_size': len(self.pattern_to_id),
            'total_usage': sum(self.pattern_usage.values()),
            'most_used_patterns': self.pattern_usage.most_common(10),
            'frequent_combinations': {
                k: dict(v.most_common(3)) 
                for k, v in list(self.pattern_combinations.items())[:5]
            }
        }
    
    def save_vocabulary(self, filepath: str):
        """Save learned vocabulary."""
        vocab_data = {
            'pattern_to_id': self.pattern_to_id,
            'id_to_pattern': {str(k): v for k, v in self.id_to_pattern.items()},
            'pattern_usage': dict(self.pattern_usage),
            'pattern_combinations': {
                k: dict(v) for k, v in self.pattern_combinations.items()
            },
            'special_tokens': self.special_tokens,
            'max_vocab_size': self.max_vocab_size
        }
        
        with open(filepath, 'w') as f:
            json.dump(vocab_data, f, indent=2)
        
        print(f"Vocabulary saved to {filepath}")
    
    def load_vocabulary(self, filepath: str):
        """Load learned vocabulary."""
        with open(filepath, 'r') as f:
            vocab_data = json.load(f)
        
        self.pattern_to_id = vocab_data['pattern_to_id']
        self.id_to_pattern = {int(k): v for k, v in vocab_data['id_to_pattern'].items()}
        self.pattern_usage = Counter(vocab_data['pattern_usage'])
        self.pattern_combinations = defaultdict(Counter)
        for k, v in vocab_data['pattern_combinations'].items():
            self.pattern_combinations[k] = Counter(v)
        self.special_tokens = vocab_data['special_tokens']
        self.max_vocab_size = vocab_data['max_vocab_size']
        
        print(f"Vocabulary loaded from {filepath}")