from typing import List
import re


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into chunks of specified size with overlap.

    Args:
        text: The text to chunk
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # If we're not at the end, try to find a good break point
        if end < len(text):
            # Look for sentence endings within the last 100 characters
            last_period = text.rfind('.', start, end)
            last_newline = text.rfind('\n', start, end)
            last_space = text.rfind(' ', start, end)

            # Use the best break point found
            break_point = max(last_period, last_newline, last_space)
            if break_point > start + chunk_size // 2:  # Only use if it's not too early
                end = break_point + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position with overlap
        start = end - overlap

        # Ensure we don't get stuck
        if start >= len(text) or start <= 0:
            break

    return chunks


def chunk_by_feedback_entries(text: str) -> List[str]:
    """
    Split text by individual feedback entries using regex patterns.
    Each feedback entry becomes a separate chunk.
    """
    # Pattern to match feedback entries (Customer Feedback #XXX)
    pattern = r'(?=Customer Feedback #\d+)'
    chunks = re.split(pattern, text)

    # Clean up chunks - remove empty strings and strip whitespace
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    return chunks


def chunk_by_categories(text: str) -> List[str]:
    """
    Split text by feedback categories, grouping related feedback together.
    """
    categories = {}
    current_category = None

    lines = text.split('\n')
    current_chunk = []

    for line in lines:
        # Check if this line contains a category
        category_match = re.search(r'Category: ([^,\n]+)', line)
        if category_match:
            # Save previous category chunk if exists
            if current_category and current_chunk:
                if current_category not in categories:
                    categories[current_category] = []
                categories[current_category].append('\n'.join(current_chunk))

            current_category = category_match.group(1).strip()
            current_chunk = [line]
        elif current_category:
            current_chunk.append(line)

    # Add the last chunk
    if current_category and current_chunk:
        if current_category not in categories:
            categories[current_category] = []
        categories[current_category].append('\n'.join(current_chunk))

    # Flatten all chunks
    all_chunks = []
    for category, chunks in categories.items():
        all_chunks.extend(chunks)

    return all_chunks


def chunk_semantic_with_overlap(text: str, max_chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Intelligent chunking that tries to split at natural boundaries.
    """
    chunks = []
    sentences = re.split(r'(?<=[.!?])\s+', text)

    current_chunk = ""
    current_size = 0

    for sentence in sentences:
        sentence_size = len(sentence)

        if current_size + sentence_size > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())

            # Start new chunk with overlap from previous chunk
            words = current_chunk.split()
            overlap_text = ' '.join(words[-50:]) if len(words) > 50 else current_chunk[-overlap:]
            current_chunk = overlap_text + " " + sentence
            current_size = len(current_chunk)
        else:
            current_chunk += " " + sentence
            current_size += sentence_size + 1

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks