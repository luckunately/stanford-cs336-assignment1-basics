import os
import multiprocessing
import concurrent.futures
from collections import defaultdict
from cs336_basics.pretokenizer import pre_tokenize
from cs336_basics.constants import UNICODE_MAX

def bpe_tokenize(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str], num_processes: int = 4, desired_num_chunks: int = 32
    ) -> tuple[dict[int, bytes],  list[tuple[bytes, bytes]]]:
    """
    Tokenize the file using byte pair encoding (BPE) with parallel processing.
    Input:
        input_path: Path to the input text file.
        vocab_size: Desired vocabulary size.
        special_tokens: List of special tokens to include in the vocabulary.
        num_processes: Number of parallel processes to use.
        desired_num_chunks: Desired number of chunks to split the file into for parallel processing.
    Output:
        A tuple containing:
            - A dictionary mapping token IDs to their corresponding byte sequences.
            - A list of tuples representing the BPE merges (pairs of byte sequences). Follow lexicographically greater pair
    """
    # Build the initial vocabulary
    vocab = {}
    
    # Add special tokens to vocabulary
    special_token_map = {}
    next_id = UNICODE_MAX
    for token in special_tokens:
        token_bytes = token.encode('utf-8')
        vocab[next_id] = token_bytes
        special_token_map[token_bytes] = next_id
        next_id += 1

    # Get tokenized chunks as separate lists
    token_chunks = pre_tokenize_parallel(input_path, num_processes=num_processes, 
                                       desired_num_chunks=desired_num_chunks, 
                                       special_tokens=special_tokens,
                                       special_token_map=special_token_map)

    # Perform BPE merges
    bpe_merges = []
    while len(vocab) < vocab_size:
        # Find the most frequent pair of byte sequences
        most_frequent_pair = find_frequent_pairs_parallel(token_chunks, vocab, num_processes=num_processes, 
                                                        special_token_map=special_token_map)
        if not most_frequent_pair:
            break
        
        # Merge the most frequent pair
        merge_pair_parallel(token_chunks, vocab, most_frequent_pair, num_processes=num_processes, next_id=next_id)
        next_id += 1
        # Convert the pair of token ids to pair of byte sequences
        bpe_merges.append((vocab[most_frequent_pair[0]], vocab[most_frequent_pair[1]]))

    return vocab, bpe_merges

def merge_pair_parallel(token_chunks: list[list[int]], 
                        vocab: dict[int, bytes], 
                        pair: tuple[int, int], 
                        num_processes: int = 4,
                        next_id: int = UNICODE_MAX) -> None:
    """Merge the pair in all token chunks and update the vocabulary"""
    new_token = vocab[pair[0]] + vocab[pair[1]]
    vocab[next_id] = new_token

    def merge_chunk(chunk):
        i = 0
        while i < len(chunk) - 1:
            if chunk[i] == pair[0] and chunk[i + 1] == pair[1]:
                chunk[i] = next_id
                del chunk[i + 1]
            else:
                i += 1
        return chunk
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
        merged_chunks = list(executor.map(merge_chunk, token_chunks))
    
    # Update token_chunks in place
    token_chunks[:] = merged_chunks

def count_pairs_in_chunk(chunk: list[int]) -> dict[tuple[int, int], int]:
    """Count pairs in a single chunk"""
    pair_freq = defaultdict(int)
    for i in range(len(chunk) - 1):
        pair = (chunk[i], chunk[i + 1])
        pair_freq[pair] += 1
    return dict(pair_freq)

def find_frequent_pairs_parallel(token_chunks: list[list[int]], vocab: dict[int, bytes], 
                                num_processes: int = 4, special_token_map: dict[bytes, int] = {}) -> tuple[int, int] | None:
    """Find the most frequent pair across all chunks using parallel processing"""
    special_token_ids = set(special_token_map.values())
    
    # Count pairs in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        chunk_pair_counts = list(executor.map(count_pairs_in_chunk, token_chunks))
    
    # Merge counts from all chunks
    total_pair_freq = defaultdict(int)
    for pair_count in chunk_pair_counts:
        for pair, count in pair_count.items():
            # Skip pairs involving special tokens
            if pair[0] in special_token_ids or pair[1] in special_token_ids:
                continue
            total_pair_freq[pair] += count
    
    if not total_pair_freq:
        return None
    
    # Sort by frequency (descending), then lexicographically by byte sequences (ascending)
    def sort_key(item):
        try: 
            pair, freq = item
            if pair[0] not in vocab:
                vocab[pair[0]] = chr(pair[0]).encode('utf-8')
            if pair[1] not in vocab:
                vocab[pair[1]] = chr(pair[1]).encode('utf-8')

            byte_seq1 = vocab[pair[0]]
            byte_seq2 = vocab[pair[1]]
            return (-freq, byte_seq1, byte_seq2)
        except KeyError:
            return (float('inf'), b'', b'')  # Place invalid pairs at the end
    
    sorted_pairs = sorted(total_pair_freq.items(), key=sort_key)
    return sorted_pairs[0][0] if sorted_pairs else None

def pre_tokenize_parallel(input_path: str | os.PathLike, num_processes: int = 4, 
                         desired_num_chunks: int = 32, special_tokens: list[str] = [],
                         special_token_map: dict[bytes, int] = {}) -> list[list[int]]:
    """Pre-tokenize file and return list of token chunks"""
    if special_tokens is []:
        special_tokens = ['<|endoftext|>']
    
    # Use the first special token for splitting
    special_split_token = special_tokens[0].encode('utf-8')
    
    # Get flat token list from existing pre_tokenize function
    token_chunks = pre_tokenize(input_path, num_processes=num_processes, 
                                desired_num_chunks=desired_num_chunks, 
                                special_split_token=special_split_token,
                                special_tokens=special_tokens
                                )
    
    return token_chunks

