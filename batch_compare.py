#!/usr/bin/env python3

import os
import re
import json
from typing import Dict, List, Tuple, Any
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import multiprocessing as mp
from tqdm import tqdm
from collections import defaultdict
import accuracy_test as at
import difflib

def extract_volume_number(filename: str) -> int:
    """Extract volume number from filename."""
    match = re.search(r'Volume\s+(\d+)', filename)
    return int(match.group(1)) if match else -1

def get_volume_pairs() -> List[Tuple[str, str]]:
    """Get pairs of regular and variant volumes."""
    data_dir = 'data'
    files = os.listdir(data_dir)
    
    # Group files by volume number
    volumes = defaultdict(dict)
    for f in files:
        if not f.endswith('.epub'):
            continue
        
        vol_num = extract_volume_number(f)
        if vol_num == -1:
            continue
            
        if '- Variant' in f:
            volumes[vol_num]['variant'] = os.path.join(data_dir, f)
        else:
            volumes[vol_num]['regular'] = os.path.join(data_dir, f)
    
    # Create pairs
    pairs = []
    for vol_num in sorted(volumes.keys()):
        if 'regular' in volumes[vol_num] and 'variant' in volumes[vol_num]:
            pairs.append((volumes[vol_num]['regular'], volumes[vol_num]['variant']))
    
    return pairs

def extract_text_from_epub(epub_path: str) -> str:
    """Extract text content from EPUB file."""
    book = epub.read_epub(epub_path)
    text_content = []
    
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            text_content.append(text)
    
    return '\n\n'.join(text_content)

def calculate_similarity_single_process(text1: str, text2: str, sample_size: int = 50, num_samples: int = 100) -> Dict[str, Any]:
    """Calculate similarity between texts without using multiprocessing."""
    text1 = at.normalize_text(text1)
    text2 = at.normalize_text(text2)
    
    samples1 = at.get_random_samples(text1, sample_size, num_samples)
    samples2 = at.get_random_samples(text2, sample_size, num_samples)
    
    window_similarities = []
    semantic_similarities = []
    
    for s1, s2 in zip(samples1, samples2):
        # Use SequenceMatcher directly for window matching
        ratio = difflib.SequenceMatcher(None, s1, s2).ratio()
        window_similarities.append(ratio)
        semantic_ratio = at.calculate_semantic_similarity(s1, s2)
        semantic_similarities.append(semantic_ratio)
    
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
        }
    }

def analyze_volume_pair(args: Tuple[str, str, int]) -> Dict[str, Any]:
    """Analyze a pair of volumes and return metrics."""
    regular_path, variant_path, vol_num = args
    
    try:
        # Extract text from both EPUBs
        regular_text = extract_text_from_epub(regular_path)
        variant_text = extract_text_from_epub(variant_path)
        
        # Analyze OCR quality
        regular_ocr = at.analyze_ocr_quality(regular_text)
        variant_ocr = at.analyze_ocr_quality(variant_text)
        
        # Analyze structure
        regular_structure = at.analyze_structure(regular_text)
        variant_structure = at.analyze_structure(variant_text)
        
        # Calculate similarity without multiprocessing
        similarity = calculate_similarity_single_process(regular_text, variant_text, sample_size=50, num_samples=100)
        
        return {
            'volume': vol_num,
            'regular': {
                'ocr': regular_ocr,
                'structure': regular_structure,
                'file_size': os.path.getsize(regular_path)
            },
            'variant': {
                'ocr': variant_ocr,
                'structure': variant_structure,
                'file_size': os.path.getsize(variant_path)
            },
            'similarity': similarity
        }
    except Exception as e:
        print(f"Error processing volume {vol_num}: {str(e)}")
        return {
            'volume': vol_num,
            'error': str(e),
            'regular': {'file_size': os.path.getsize(regular_path)},
            'variant': {'file_size': os.path.getsize(variant_path)}
        }

def main():
    print("Finding volume pairs...")
    pairs = get_volume_pairs()
    print(f"Found {len(pairs)} volume pairs")
    
    # Prepare arguments for parallel processing
    args = [(reg, var, extract_volume_number(reg)) for reg, var in pairs]
    
    print("\nAnalyzing volumes in parallel...")
    results = []
    with mp.Pool(mp.cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(analyze_volume_pair, args),
                         total=len(args),
                         desc="Processing volumes"):
            if result and 'error' not in result:  # Only include successful results
                results.append(result)
    
    # Sort results by volume number
    results.sort(key=lambda x: x['volume'])
    
    # Calculate aggregate statistics
    total_volumes = len(results)
    if total_volumes == 0:
        print("\nNo volumes were processed successfully.")
        return
        
    avg_similarity = sum(r['similarity']['window']['average'] for r in results) / total_volumes
    avg_ocr_quality_regular = sum(r['regular']['ocr']['dictionary_ratio'] for r in results) / total_volumes
    avg_ocr_quality_variant = sum(r['variant']['ocr']['dictionary_ratio'] for r in results) / total_volumes
    
    print("\n=== HOLISTIC ANALYSIS RESULTS ===")
    print(f"\nProcessed {total_volumes} volumes successfully")
    print(f"Average similarity across all volumes: {avg_similarity:.2f}%")
    print(f"Average OCR quality (regular): {avg_ocr_quality_regular*100:.2f}%")
    print(f"Average OCR quality (variant): {avg_ocr_quality_variant*100:.2f}%")
    
    print("\nDetailed Results by Volume:")
    for r in results:
        vol = r['volume']
        sim = r['similarity']['window']['average']
        reg_ocr = r['regular']['ocr']['dictionary_ratio'] * 100
        var_ocr = r['variant']['ocr']['dictionary_ratio'] * 100
        print(f"\nVolume {vol}:")
        print(f"- Similarity: {sim:.2f}%")
        print(f"- Regular OCR Quality: {reg_ocr:.2f}%")
        print(f"- Variant OCR Quality: {var_ocr:.2f}%")
        print(f"- Words (Regular/Variant): {r['regular']['structure']['words']}/{r['variant']['structure']['words']}")
    
    # Save detailed results to file
    print("\nSaving detailed results to batch_analysis_results.json...")
    with open('batch_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main() 