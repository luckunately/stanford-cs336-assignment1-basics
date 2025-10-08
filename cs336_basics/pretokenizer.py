import os
from typing import BinaryIO
import multiprocessing
import concurrent.futures
import regex as re

def revert_regex_tokenize(tokens: list[int], split_special_token: bytes = b'<|endoftext|>') -> bytes:
    chars = []
    for token in tokens:
        if token == 256:  # special token
            chars.append(split_special_token)
        else:
            chars.append(bytes([token]))  # Convert ASCII value back to character
    return b''.join(chars)

def regex_tokenize(text: bytes, split_special_token: bytes = b'<|endoftext|>') -> list[int]:
    """
    Tokenizes the input bytes using a regex-based approach, splitting on a special token.

    Args:
        text (bytes): The input text to tokenize, as a bytes object.
        split_special_token (bytes, optional): The special token used to split the text. 
            Defaults to b'<|endoftext|>'.

    Returns:
        list[int]: A list of integer tokens. Each token is the UTF-8 value of a character 
            from the regex-matched substrings, with the special token represented by 256.
            The special token is appended after each split part, except possibly at the end 
            if the input does not end with the special token.

    Notes:
        - The function splits the input bytes on the special token, then applies a regex 
          to each part to extract tokens.
        - Each regex-matched substring is converted to a sequence of UTF-8 integer values.
        - The special token (represented by 256) is appended after each part.
        - If the input does not end with the special token, the final 256 is removed.
    """
    # split the text by the split_special_token first
    text_parts = text.split(split_special_token)
    
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pattern = re.compile(PAT)

    def process_part(part):
        tokens = []
        if part:
            found_tokens = pattern.findall(part.decode('utf-8', errors='ignore'))
            for token in found_tokens:
                tokens.extend(ord(char) for char in token)
        tokens.append(256)  # Always append special token after each part
        return tokens

    # Use ThreadPoolExecutor for IO-bound (decoding) or ProcessPoolExecutor for CPU-bound (regex)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_part, text_parts))
    
    # Flatten the list of lists
    list_tokens = [item for sublist in results for item in sublist]
    
    # if text did not end with split_special_token, remove the last appended 256
    if not text.endswith(split_special_token) and list_tokens and list_tokens[-1] == 256:
        list_tokens.pop()
    
    # flatten the list and convert to int
    return list_tokens

def tokenize_chunk(file: str, split_special_token: bytes, start_end: tuple[int, int]) -> list[int]:
    start, end = start_end
    with open(file, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)
        return regex_tokenize(chunk, split_special_token)

def pre_tokenize(file: str | os.PathLike, num_processes: int = 4, desired_num_chunks: int = 32, special_split_token : bytes = b'<|endoftext|>') -> list[int]:
    # Find chunk boundaries
    with open(file, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks, special_split_token)
    
    # Tokenize each chunk in parallel

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        length = len(boundaries) - 1
        files = [file] * length
        split_tokens = [special_split_token] * length
        results = list(executor.map(tokenize_chunk, files, split_tokens, 
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