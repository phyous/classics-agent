# Harvard Classics Text Analysis Tools

Tools for analyzing and comparing different versions of Harvard Classics texts, with a focus on OCR quality assessment and text comparison.

## Data Source
EPUB versions available at: https://archive.org/compress/harvard-classics-all-51-volumes/formats=EPUB&file=/harvard-classics-all-51-volumes.zip

## Setup Instructions

1. **Environment Setup**
```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install tqdm
```

2. **Directory Structure**
```
classics-agent/
├── data/                    # Store EPUB files here
├── accuracy_test.py         # Main analysis script
└── README.md
```

3. **Data Preparation**
- Download the Harvard Classics EPUB files from the archive.org link
- Extract the desired volume to analyze
- Convert EPUB to markdown using the epub_reader.py script
- Save the archive.org text version for comparison

## Usage Examples

### 1. Compare EPUB-derived text with Archive.org text

```bash
# Basic comparison of two text versions
python accuracy_test.py
```

This will:
- Load both text versions
- Analyze OCR quality metrics
- Show structural analysis
- Provide similarity metrics
- Display 20 text comparison examples

### 2. Script Output Explanation

The script provides several types of analysis:

a) **OCR Quality Metrics**
- Total word count
- Suspicious word detection
- Dictionary word ratio
- Hyphenation issues
- Example errors

b) **Structural Analysis**
- Paragraph count
- Sentence count
- Word count
- Table of contents detection
- Essay detection
- Unique word count

c) **Similarity Analysis**
- Window match metrics (exact matches)
- Semantic similarity metrics
- Text comparison examples

### 3. Example Output

```
=== OCR QUALITY ANALYSIS ===

Our Version:
- Total Words: 133,322
- Suspicious Words: 13,587 (10.19%)
- Dictionary Word Ratio: 7.47%
- Hyphenation Issues: 1

Archive Version:
- Total Words: 142,156
- Suspicious Words: 15,843 (11.14%)
- Dictionary Word Ratio: 7.19%
- Hyphenation Issues: 1,314
```

## Performance Notes

- The script uses multiprocessing for faster analysis
- Default sample size: 50 words
- Default number of samples: 200
- Adjustable parameters in the script:
  - `sample_size`: Length of text segments to compare
  - `num_samples`: Number of random samples to analyze
  - `window_size`: Size of sliding window for matching

## Contributing

Feel free to submit issues and enhancement requests!
