import os
from typing import BinaryIO
import multiprocessing
import concurrent.futures
import regex as re
from cs336_basics.constants import UNICODE_MAX

def revert_regex_tokenize(tokens: list[int], special_tokens: list[str] = None) -> bytes:
    """Revert tokens back to bytes, supporting multiple special tokens"""
    if special_tokens is None:
        special_tokens = ['<|endoftext|>']
    
    # Create mapping from token IDs to special tokens
    special_token_map = {}
    for i, token in enumerate(special_tokens):
        special_token_map[256 + i] = token.encode('utf-8')
    
    chars = []
    for token in tokens:
        if token in special_token_map:
            chars.append(special_token_map[token])
        else:
            chars.append(bytes([token]))  # Convert ASCII value back to character
    return b''.join(chars)

def regex_tokenize(text: bytes, special_tokens: list[str], special_token_id: int = UNICODE_MAX) -> list[list[int]]:
    """
    Tokenizes the input bytes using a regex-based approach, splitting on special tokens.

    Args:
        text (bytes): The input text to tokenize, as a bytes object.
        special_tokens (list[str], optional): List of special tokens used to split the text. 
            Defaults to ['<|endoftext|>'].

    Returns:
        list[int]: A list of integer tokens. Each token is the UTF-8 value of a character 
            from the regex-matched substrings, with special tokens represented by 256+index.
    """
    if special_tokens is []:
        raise ValueError("special_tokens must be a non-empty list")
        
    
    # Use the first special token for splitting (backward compatibility)
    split_special_token = special_tokens[0].encode('utf-8')
    
    # split the text by the split_special_token first
    text_parts = text.split(split_special_token)
    
    # recurse for the next special tokens if any
    if len(special_tokens) > 1:
        # Spawn a thread for each part in text_parts
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(
            lambda part: regex_tokenize(part, special_tokens[1:], special_token_id + 1),
            text_parts
            ))
        # Flatten the results
        list_tokens = [item for sublist in results for item in sublist]
        
        # if text did not end with split_special_token, remove the last appended special token
        if not text.endswith(split_special_token) and list_tokens and list_tokens[-1] == special_token_id:
            list_tokens.pop()
        return list_tokens
    
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pattern = re.compile(PAT)

    def process_part(part):
        tokens = []
        if part:
            found_token_iterator = pattern.finditer(part.decode('utf-8', errors='ignore'))
            for token in found_token_iterator:
                value = [ord(char) for char in token.group()]
                tokens.extend(value)
        tokens.append(special_token_id)  # Always append special token after each part
        return tokens

    # Use ThreadPoolExecutor for IO-bound (decoding) or ProcessPoolExecutor for CPU-bound (regex)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_part, text_parts))
    
    # if text did not end with split_special_token, remove the last appended special token
    if not text.endswith(split_special_token) and results and results[-1][-1] == special_token_id:
        results[-1].pop()

    return results

def tokenize_chunk(file: str, special_tokens: list[str], start_end: tuple[int, int]) -> list[list[int]]:
    start, end = start_end
    with open(file, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)
        return regex_tokenize(chunk, special_tokens)

def pre_tokenize(file: str | os.PathLike, num_processes: int = 4, desired_num_chunks: int = 32, 
                special_split_token: bytes = b'<|endoftext|>', special_tokens: list[str] = []) -> list[list[int]]:
    """Pre-tokenize with support for multiple special tokens"""
    # Find chunk boundaries
    with open(file, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks, special_split_token)
    
    # Tokenize each chunk in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        length = len(boundaries) - 1
        files = [file] * length
        special_token_lists = [special_tokens] * length
        results = list(executor.map(tokenize_chunk, files, special_token_lists, 
                                    [(boundaries[i], boundaries[i+1]) for i in range(length)]))

    # Flatten the list of lists
    return [item for sublist in results for item in sublist]

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))