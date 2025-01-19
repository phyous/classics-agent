#!/usr/bin/env python3

import difflib
import re
from typing import List, Tuple, Dict
import os
import random
from collections import Counter
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

def load_file(filepath: str) -> str:
    """Load and clean text from a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def normalize_text(text: str) -> str:
    """Normalize text for comparison by removing all formatting differences."""
    # Convert to lowercase
    text = text.lower()
    
    # Replace common character variations
    replacements = {
        ''': "'",
        '"': '"',
        '"': '"',
        '—': '-',
        '–': '-',
        '`': "'",
        ''': "'"
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove all punctuation and special characters
    text = re.sub(r'[^\w\s-]', '', text)
    
    # Remove all numbers
    text = re.sub(r'\d+', '', text)
    
    # Convert all whitespace (including newlines, tabs) to single spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common formatting words that don't affect meaning
    stop_words = {'chapter', 'page', 'volume', 'book', 'section', 'part'}
    words = text.split()
    words = [w for w in words if w not in stop_words]
    
    return ' '.join(words).strip()

def get_random_samples(text: str, sample_size: int = 50, num_samples: int = 200) -> List[str]:
    """Get random samples of text efficiently."""
    words = text.split()
    if len(words) <= sample_size:
        return [text]
    
    max_start = len(words) - sample_size
    samples = []
    
    # Generate all random indices at once
    start_indices = [random.randint(0, max_start) for _ in range(num_samples)]
    
    for start_idx in start_indices:
        sample = ' '.join(words[start_idx:start_idx + sample_size])
        samples.append(sample)
    
    return samples

def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity using fast set operations."""
    # Convert to sets for faster operations
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    # Use set operations instead of Counter
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0

def find_best_window_match_with_text(sample: str, text: str, window_size: int = 150) -> Tuple[float, str]:
    """Find the best matching window in the text and return both ratio and matching text."""
    words = text.split()
    sample_words = sample.split()
    
    if len(words) < len(sample_words):
        return (len(set(words) & set(sample_words)) / len(set(words) | set(sample_words)), ' '.join(words))
    
    sample_set = frozenset(sample_words)
    sample_len = len(sample_words)
    
    best_ratio = 0
    best_window = ''
    step_size = max(1, sample_len // 2)
    
    for i in range(0, len(words) - sample_len + 1, step_size):
        window = words[i:i + sample_len]
        window_set = frozenset(window)
        ratio = len(sample_set & window_set) / len(sample_set | window_set)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_window = ' '.join(window)
            if best_ratio > 0.8:
                break
    
    return best_ratio, best_window

def process_sample_chunk(args):
    """Process a chunk of samples in parallel and return text examples."""
    samples1, text2 = args
    results = []
    for s1 in samples1:
        ratio, matching_text = find_best_window_match_with_text(s1, text2)
        semantic_ratio = calculate_semantic_similarity(s1, text2)
        results.append((ratio, semantic_ratio, s1, matching_text))
    return results

def calculate_similarity(text1: str, text2: str, sample_size: int = 50, num_samples: int = 200) -> Dict:
    """Calculate similarity between texts and collect example differences."""
    text1 = normalize_text(text1)
    text2 = normalize_text(text2)
    
    samples1 = get_random_samples(text1, sample_size, num_samples)
    
    num_cores = mp.cpu_count()
    chunk_size = max(1, num_samples // (num_cores * 4))
    sample_chunks = [samples1[i:i + chunk_size] for i in range(0, len(samples1), chunk_size)]
    
    pool_args = [(chunk, text2) for chunk in sample_chunks]
    
    window_similarities = []
    semantic_similarities = []
    examples = []  # Store text examples
    
    print(f"\nProcessing {num_samples} samples using {num_cores} cores...")
    with mp.Pool(num_cores) as pool:
        for chunk_results in tqdm(pool.imap_unordered(process_sample_chunk, pool_args),
                                total=len(pool_args),
                                desc="Processing chunks"):
            for window_ratio, semantic_ratio, source_text, match_text in chunk_results:
                window_similarities.append(window_ratio)
                semantic_similarities.append(semantic_ratio)
                examples.append((window_ratio, source_text, match_text))
    
    # Sort examples by similarity ratio and select diverse samples
    examples.sort(key=lambda x: x[0])
    step = len(examples) // 20
    selected_examples = examples[::step][:20]  # Take 20 evenly distributed samples
    
    sorted_window = sorted(window_similarities)
    sorted_semantic = sorted(semantic_similarities)
    mid = len(sorted_window) // 2
    quarter = len(sorted_window) // 4
    
    return {
        'window': {
            'average': sum(window_similarities) / len(window_similarities) * 100,
            'median': sorted_window[mid] * 100,
            'min': min(window_similarities) * 100,
            'max': max(window_similarities) * 100,
            'q1': sorted_window[quarter] * 100,
            'q3': sorted_window[3 * quarter] * 100
        },
        'semantic': {
            'average': sum(semantic_similarities) / len(semantic_similarities) * 100,
            'median': sorted_semantic[mid] * 100,
            'min': min(semantic_similarities) * 100,
            'max': max(semantic_similarities) * 100,
            'q1': sorted_semantic[quarter] * 100,
            'q3': sorted_semantic[3 * quarter] * 100
        },
        'num_samples': len(window_similarities),
        'examples': selected_examples
    }

def analyze_structure(text: str) -> Dict:
    """Analyze structural elements of the text."""
    # Store original and normalized versions
    normalized = normalize_text(text)
    
    results = {
        'paragraphs': len(re.findall(r'\n\s*\n', text)),
        'sentences': len(re.findall(r'[.!?]+', text)),
        'words': len(normalized.split()),  # Count normalized words
        'has_toc': bool(re.search(r'CONTENTS|Table of Contents', text, re.IGNORECASE)),
        'has_essays': bool(re.search(r'Essays.*Civil and Moral', text, re.IGNORECASE)),
        'unique_words': len(set(normalized.split()))  # Count unique normalized words
    }
    return results

def detect_ocr_errors(text: str) -> List[str]:
    """Detect likely OCR errors in text."""
    # Common OCR error patterns
    suspicious_patterns = [
        r'\b[a-z]+[A-Z]+[a-z]+\b',  # Mixed case within word
        r'\b[a-z]{1,2}\b',          # Very short words (likely fragments)
        r'[a-z][0-9][a-z]',         # Numbers mixed with letters
        r'[^\s\w\-\']',             # Unusual characters
        r'\b[bcdefghijklmnopqrstuvwxyz]{8,}\b',  # Long words without vowels
        r'\b(th[eo]|tha|wh[eo]|wha)\b',  # Common article fragments
        r'[aeiou]{3,}',             # Too many consecutive vowels
        r'[bcdfghjklmnpqrstvwxz]{4,}',  # Too many consecutive consonants
    ]
    
    errors = []
    words = text.split()
    
    for word in words:
        for pattern in suspicious_patterns:
            if re.search(pattern, word):
                errors.append(word)
                break
    
    return list(set(errors))  # Remove duplicates

def analyze_ocr_quality(text: str) -> Dict:
    """Analyze OCR quality metrics."""
    words = text.split()
    total_words = len(words)
    
    # Find suspicious words
    suspicious_words = detect_ocr_errors(text)
    
    # Calculate dictionary word ratio using a basic English word check
    english_words = set(word.lower() for word in words if word.isalpha() and len(word) > 1)
    dictionary_ratio = len(english_words) / total_words if total_words > 0 else 0
    
    # Analyze hyphenation consistency
    hyphenated = len(re.findall(r'\w+\-\w+', text))
    suspicious_hyphens = len(re.findall(r'\w+\-\s+\w+', text))  # Hyphen with space after
    
    return {
        'total_words': total_words,
        'suspicious_words': len(suspicious_words),
        'suspicious_word_ratio': len(suspicious_words) / total_words if total_words > 0 else 0,
        'dictionary_ratio': dictionary_ratio,
        'hyphenation_issues': suspicious_hyphens,
        'example_errors': sorted(suspicious_words)[:10]  # Show first 10 examples
    }

def main():
    print("Loading texts...")
    our_text = load_file(os.path.expanduser("~/Desktop/test/volume_3.md"))
    archive_text = load_file("archive_text.txt")
    
    print("\nAnalyzing document structure...")
    with tqdm(total=2, desc="Analyzing structure") as pbar:
        our_structure = analyze_structure(our_text)
        pbar.update(1)
        archive_structure = analyze_structure(archive_text)
        pbar.update(1)
    
    print("\nAnalyzing OCR quality...")
    our_ocr = analyze_ocr_quality(our_text)
    archive_ocr = analyze_ocr_quality(archive_text)
    
    print("\n=== OCR QUALITY ANALYSIS ===")
    print("\nOur Version:")
    print(f"- Total Words: {our_ocr['total_words']}")
    print(f"- Suspicious Words: {our_ocr['suspicious_words']} ({our_ocr['suspicious_word_ratio']*100:.2f}%)")
    print(f"- Dictionary Word Ratio: {our_ocr['dictionary_ratio']*100:.2f}%")
    print(f"- Hyphenation Issues: {our_ocr['hyphenation_issues']}")
    print("- Example Errors:", ', '.join(our_ocr['example_errors']))
    
    print("\nArchive Version:")
    print(f"- Total Words: {archive_ocr['total_words']}")
    print(f"- Suspicious Words: {archive_ocr['suspicious_words']} ({archive_ocr['suspicious_word_ratio']*100:.2f}%)")
    print(f"- Dictionary Word Ratio: {archive_ocr['dictionary_ratio']*100:.2f}%")
    print(f"- Hyphenation Issues: {archive_ocr['hyphenation_issues']}")
    print("- Example Errors:", ', '.join(archive_ocr['example_errors']))
    
    print("\nPerforming random sampling analysis...")
    similarity_results = calculate_similarity(our_text, archive_text, sample_size=50, num_samples=200)
    
    print("\n=== ACCURACY ANALYSIS RESULTS ===")
    
    print("\nStructural Analysis:")
    print("\nOur Version:")
    for key, value in our_structure.items():
        print(f"- {key}: {value}")
    
    print("\nArchive Version:")
    for key, value in archive_structure.items():
        print(f"- {key}: {value}")
    
    print("\nRandom Sampling Analysis:")
    print(f"Number of samples: {similarity_results['num_samples']}")
    print(f"Sample size: 50 words")
    
    print("\nWindow Match Metrics:")
    for key, value in similarity_results['window'].items():
        if key != 'num_samples':
            print(f"- {key.capitalize():8}: {value:.2f}%")
    
    print("\nSemantic Similarity Metrics:")
    for key, value in similarity_results['semantic'].items():
        if key != 'num_samples':
            print(f"- {key.capitalize():8}: {value:.2f}%")
    
    word_diff_pct = abs(our_structure['words'] - archive_structure['words']) / max(our_structure['words'], archive_structure['words']) * 100
    unique_word_diff_pct = abs(our_structure['unique_words'] - archive_structure['unique_words']) / max(our_structure['unique_words'], archive_structure['unique_words']) * 100
    
    print(f"\nDifferences:")
    print(f"- Total Word Count: {word_diff_pct:.2f}%")
    print(f"- Unique Word Count: {unique_word_diff_pct:.2f}%")
    
    print("\n=== TEXT COMPARISON EXAMPLES ===")
    print("\nShowing 20 examples with varying similarity levels:")
    for i, (ratio, source, match) in enumerate(similarity_results['examples'], 1):
        print(f"\nExample {i} (Similarity: {ratio*100:.2f}%):")
        print(f"Our Version:     {source}")
        print(f"Archive Version: {match}")
        print("-" * 80)

if __name__ == "__main__":
    main() 