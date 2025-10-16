"""
Text Processing Utilities for RAG (Retrieval-Augmented Generation) System

This module contains various text chunking strategies used to break down large documents
into smaller, manageable pieces that can be effectively indexed and searched in a vector database.

Chunking is crucial in RAG systems because:
- Large documents exceed token limits of language models
- Smaller chunks provide more precise search results
- Overlap between chunks maintains context continuity
"""

from typing import List
import re


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Basic text chunking with intelligent boundary detection.

    This function splits text into chunks of approximately 'chunk_size' characters,
    but tries to break at natural boundaries (sentences, lines, words) rather than
    cutting words in half. It also adds overlap between chunks to maintain context.

    Args:
        text (str): The input text to be chunked
        chunk_size (int): Maximum characters per chunk (default: 1000)
        overlap (int): Characters to overlap between consecutive chunks (default: 200)

    Returns:
        List[str]: List of text chunks

    Example:
        >>> text = "This is sentence one. This is sentence two. This is sentence three."
        >>> chunks = chunk_text(text, chunk_size=20, overlap=5)
        >>> print(len(chunks))  # Multiple chunks with overlap
    """
    # If the entire text fits in one chunk, return it as-is
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0  # Starting position for current chunk

    # Continue until we've processed the entire text
    while start < len(text):
        # Calculate where this chunk should end
        end = start + chunk_size

        # If we're not at the end of the text, try to find a better break point
        if end < len(text):
            # Look for natural break points within the last 100 characters of this chunk
            # Priority: periods (sentence ends) > newlines > spaces (word boundaries)
            last_period = text.rfind('.', start, end)      # Sentence boundary
            last_newline = text.rfind('\n', start, end)    # Line boundary
            last_space = text.rfind(' ', start, end)       # Word boundary

            # Choose the best break point found (rightmost = most recent)
            break_point = max(last_period, last_newline, last_space)

            # Only use this break point if it's not too early in the chunk
            # (avoids breaking too soon and creating very small chunks)
            if break_point > start + chunk_size // 2:
                end = break_point + 1  # Include the break character

        # Extract the chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)

        # Move start position for next chunk, accounting for overlap
        start = end - overlap

        # Safety check to prevent infinite loops
        if start >= len(text) or start <= 0:
            break

    return chunks


def chunk_by_feedback_entries(text: str) -> List[str]:
    """
    Document-structure aware chunking for customer feedback databases.

    This function splits text based on document structure rather than character count.
    It looks for patterns like "Customer Feedback #001", "#002", etc. and splits
    the text at these boundaries. This ensures each feedback entry remains intact.

    This is particularly useful for:
    - Customer feedback databases
    - Support ticket collections
    - Structured document repositories

    Args:
        text (str): The input text containing multiple feedback entries

    Returns:
        List[str]: List of individual feedback entries as separate chunks

    Example:
        >>> text = "Customer Feedback #001...Customer Feedback #002..."
        >>> chunks = chunk_by_feedback_entries(text)
        >>> print(len(chunks))  # Number of feedback entries
    """
    # Regex pattern explanation:
    # (?=...) is a positive lookahead - matches without including the text
    # Customer Feedback #\d+ matches "Customer Feedback #" followed by digits
    # This splits the text BEFORE each feedback entry marker
    pattern = r'(?=Customer Feedback #\d+)'

    # Split the text using the pattern
    chunks = re.split(pattern, text)

    # Clean up the chunks: remove empty strings and strip whitespace
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    return chunks


def chunk_by_categories(text: str) -> List[str]:
    """
    Semantic chunking based on content categories/topics.

    This advanced chunking strategy groups related content together based on
    category labels found in the text. For example, all "Bug Reports" might be
    grouped into one chunk, all "Feature Requests" into another.

    This is useful when you want to:
    - Group similar content for better retrieval
    - Maintain topic coherence in chunks
    - Reduce the number of chunks while preserving meaning

    The function looks for lines containing "Category: [CategoryName]"
    and groups all subsequent lines until the next category.

    Args:
        text (str): The input text with category labels

    Returns:
        List[str]: List of chunks, each containing all content for one category

    Example:
        >>> text = "Category: Bug Report\\nIssue description...\\nCategory: Feature Request\\n..."
        >>> chunks = chunk_by_categories(text)
        >>> print(len(chunks))  # Number of unique categories found
    """
    # Dictionary to store content for each category
    categories = {}
    # Track which category we're currently processing
    current_category = None
    # Temporary storage for lines belonging to current category
    current_chunk = []

    # Process text line by line
    lines = text.split('\n')

    for line in lines:
        # Check if this line contains a category marker
        # Regex looks for "Category: " followed by text until comma or newline
        category_match = re.search(r'Category: ([^,\n]+)', line)

        if category_match:
            # Found a new category! Save the previous category's content
            if current_category and current_chunk:
                # Initialize category list if it doesn't exist
                if current_category not in categories:
                    categories[current_category] = []
                # Add the accumulated content as a single chunk
                categories[current_category].append('\n'.join(current_chunk))

            # Start processing the new category
            current_category = category_match.group(1).strip()
            current_chunk = [line]  # Start new chunk with this category line

        elif current_category:
            # This line belongs to the current category
            current_chunk.append(line)

    # Don't forget to save the last category's content
    if current_category and current_chunk:
        if current_category not in categories:
            categories[current_category] = []
        categories[current_category].append('\n'.join(current_chunk))

    # Flatten all category chunks into a single list
    all_chunks = []
    for category, chunks in categories.items():
        all_chunks.extend(chunks)

    return all_chunks


def chunk_semantic_with_overlap(text: str, max_chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Advanced semantic chunking with sentence awareness and intelligent overlap.

    This is the most sophisticated chunking strategy that:
    1. Splits text into individual sentences first
    2. Groups sentences into chunks without exceeding size limits
    3. Adds meaningful overlap using actual words (not just characters)

    Benefits:
    - Respects natural language boundaries (sentences)
    - Maintains semantic coherence within chunks
    - Provides better context through word-based overlap
    - Avoids cutting sentences in half

    Args:
        text (str): The input text to be chunked
        max_chunk_size (int): Maximum characters per chunk (default: 1000)
        overlap (int): Characters for overlap, used as fallback for word-based overlap

    Returns:
        List[str]: List of semantically coherent chunks with overlap

    Example:
        >>> text = "First sentence. Second sentence. Third sentence."
        >>> chunks = chunk_semantic_with_overlap(text, max_chunk_size=30)
        >>> # Chunks will respect sentence boundaries
    """
    chunks = []

    # Split text into sentences using regex
    # (?<=[.!?]) is a positive lookbehind for sentence endings
    # \s+ matches one or more whitespace characters
    sentences = re.split(r'(?<=[.!?])\s+', text)

    current_chunk = ""  # Current chunk being built
    current_size = 0    # Character count of current chunk

    for sentence in sentences:
        sentence_size = len(sentence)

        # Check if adding this sentence would exceed the chunk size
        if current_size + sentence_size > max_chunk_size and current_chunk:
            # Current chunk is full, save it
            chunks.append(current_chunk.strip())

            # Start new chunk with overlap from the previous chunk
            # Try to overlap using actual words for better context
            words = current_chunk.split()

            # Take the last 50 words if available, otherwise use character-based overlap
            if len(words) > 50:
                overlap_text = ' '.join(words[-50:])
            else:
                # Fallback to character-based overlap
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk

            # Start new chunk with overlap + current sentence
            current_chunk = overlap_text + " " + sentence
            current_size = len(current_chunk)

        else:
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            current_size += sentence_size + 1  # +1 for space

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks