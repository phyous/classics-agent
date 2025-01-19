#!/usr/bin/env python3

import argparse
import sys
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import html2text
import os
import re

def debug_print(msg):
    """Print debug information."""
    print(f"DEBUG: {msg}", file=sys.stderr)

def clean_text(text):
    """Clean up text by removing extra whitespace and empty lines."""
    if not text:
        return ""
    lines = []
    for line in text.split('\n'):
        line = re.sub(r'\s+', ' ', line.strip())
        if line and not line.isspace() and line != '#':
            lines.append(line)
    return '\n\n'.join(lines)

def convert_html_to_text(html_content):
    """Convert HTML content to plain text."""
    if not html_content:
        return ""
    debug_print(f"Converting HTML content (length: {len(html_content)})")
    soup = BeautifulSoup(html_content, 'html.parser')
    # Remove unwanted elements
    for element in soup(['script', 'style', 'head', 'title', 'meta', '[document]']):
        element.decompose()
    # Get text with proper spacing
    text = ' '.join(soup.stripped_strings)
    debug_print(f"Extracted text length: {len(text)}")
    return clean_text(text)

def convert_html_to_markdown(html_content):
    """Convert HTML content to markdown."""
    if not html_content:
        return ""
    debug_print(f"Converting HTML to markdown (length: {len(html_content)})")
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.body_width = 0
    h.ignore_images = True
    h.ignore_tables = False
    h.ignore_emphasis = False
    text = h.handle(html_content)
    debug_print(f"Converted markdown length: {len(text)}")
    return clean_text(text)

def get_content_items(book):
    """Get content items in the correct order."""
    items = []
    debug_print("Getting content items...")
    
    # Try spine first
    if book.spine:
        debug_print(f"Found {len(book.spine)} spine items")
        for item_id, linear in book.spine:
            debug_print(f"Processing spine item: {item_id}")
            item = book.get_item_with_id(item_id)
            if item:
                debug_print(f"Item type: {item.get_type()}, Name: {item.get_name()}")
                # Accept any HTML content
                if item.get_name().endswith('.html') or item.get_name().endswith('.xhtml'):
                    items.append(item)
    
    # If no items found in spine, try all items
    if not items:
        debug_print("No spine items found, trying all items")
        for item in book.get_items():
            if item.get_name().endswith('.html') or item.get_name().endswith('.xhtml'):
                debug_print(f"Found HTML item: {item.get_name()}")
                items.append(item)
    
    debug_print(f"Found {len(items)} content items")
    return items

def read_epub(epub_path, num_lines=None, output_format='markdown', output_file=None):
    """Read and process an EPUB file."""
    try:
        debug_print(f"Processing EPUB file: {epub_path}")
        if not os.path.exists(epub_path):
            raise FileNotFoundError(f"EPUB file not found: {epub_path}")
            
        book = epub.read_epub(epub_path)
        debug_print("Successfully loaded EPUB file")
        
        sections = []
        
        # Extract metadata
        title = book.get_metadata('DC', 'title')
        creator = book.get_metadata('DC', 'creator')
        debug_print(f"Metadata - Title: {title}, Creator: {creator}")
        
        if title:
            sections.append(f"# {title[0][0]}")
        if creator:
            sections.append(f"By {creator[0][0]}")
        
        # Process content
        for item in get_content_items(book):
            try:
                debug_print(f"Processing item: {item.get_name()}")
                content = item.get_content().decode('utf-8')
                if not content.strip():
                    debug_print("Empty content, skipping")
                    continue
                
                if output_format == 'plain-text':
                    text = convert_html_to_text(content)
                else:  # markdown
                    text = convert_html_to_markdown(content)
                
                if text.strip():
                    debug_print(f"Adding content section (length: {len(text)})")
                    sections.append(text)
                else:
                    debug_print("No text extracted from content")
            
            except Exception as e:
                debug_print(f"Error processing item: {str(e)}")
                continue
        
        debug_print(f"Total sections: {len(sections)}")
        
        # Combine all sections
        result = '\n\n'.join(sections)
        debug_print(f"Final content length: {len(result)}")
        
        # Validate output
        if not result.strip():
            raise ValueError("No content was extracted from the EPUB file")
        
        # Limit lines if specified
        if num_lines is not None:
            result = '\n'.join(result.split('\n')[:num_lines])
        
        # Handle output
        if output_file:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"Successfully wrote content to {output_file}")
        else:
            print(result)
        
        return result
    
    except Exception as e:
        print(f"Error processing EPUB file: {str(e)}", file=sys.stderr)
        if not output_file:
            sys.exit(1)
        return None

def main():
    parser = argparse.ArgumentParser(description='Read and format EPUB files')
    parser.add_argument('--batch', action='store_true', help='Process all EPUB files in data directory')
    parser.add_argument('epub_file', nargs='?', help='Path to the EPUB file')
    parser.add_argument('-n', '--num-lines', type=int, help='Number of lines to output')
    parser.add_argument('-f', '--format', choices=['plain-text', 'markdown'], 
                        default='markdown', help='Output format (default: markdown)')
    parser.add_argument('-o', '--output-file', help='Output file path')
    
    args = parser.parse_args()
    
    if args.batch:
        data_dir = 'data'
        if not os.path.exists(data_dir):
            print(f"Error: {data_dir} directory not found", file=sys.stderr)
            sys.exit(1)
            
        files = [f for f in os.listdir(data_dir) if f.endswith('.epub')]
        total_files = len(files)
        print(f"Found {total_files} EPUB files to process")
        
        for i, epub_file in enumerate(files, 1):
            input_path = os.path.join(data_dir, epub_file)
            output_file = os.path.join(data_dir, epub_file.replace('.epub', '.md'))
            print(f"\nProcessing file {i}/{total_files}: {epub_file}")
            try:
                read_epub(input_path, args.num_lines, args.format, output_file)
            except Exception as e:
                print(f"Error processing {epub_file}: {str(e)}", file=sys.stderr)
                continue
    else:
        if not args.epub_file:
            parser.error("epub_file is required when not using --batch mode")
        read_epub(args.epub_file, args.num_lines, args.format, args.output_file)

if __name__ == '__main__':
    main()
