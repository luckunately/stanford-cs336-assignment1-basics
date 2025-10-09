import os
from collections import Counter
import concurrent.futures
from typing import Dict, List, Tuple, Set
from cs336_basics.pretokenizer import pre_tokenize

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
    for i in range(256):
        vocab[i] = bytes([i])
    # for now, only support b'<|endoftext|>'
    assert '<|endoftext|>'in special_tokens, f"Currently only support <|endoftext|> as special token, you provided: {special_tokens}"
    special_split_token = b'<|endoftext|>'
    special_encoding = 256  # Assign ID 256 to the special token
    
    vocab[special_encoding] = b'<|endoftext|>'  # Add special token

    initial_tokens = pre_tokenize(input_path, num_processes=num_processes, desired_num_chunks=desired_num_chunks, special_split_token=special_split_token)

    # Perform BPE merges
    bpe_merges = []
    while len(vocab) < vocab_size:
        # Find the most frequent pair of byte sequences
        pairs = find_frequent_pairs(initial_tokens, vocab, num_processes=num_processes, special_encoding=special_encoding)
        if not pairs:
            break
        # Merge the most frequent pair
        merge_pair(initial_tokens, vocab, pairs[0])
        # covert the pair of token ids to pair of byte sequences
        bpe_merges.append((vocab[pairs[0][0]], vocab[pairs[0][1]]))

    return vocab, bpe_merges

def merge_pair(token_arrays: list[int], vocab: dict[int, bytes], pair: tuple[int, int]) -> None:
    # Merge the pair in the token arrays and update the vocabulary
    new_token = vocab[pair[0]] + vocab[pair[1]]
    i = len(token_arrays) - 2
    while i >= 0:
        if token_arrays[i] == pair[0] and token_arrays[i + 1] == pair[1]:
            token_arrays[i] = len(vocab)
            del token_arrays[i + 1]
            i -= 1
        i -= 1
    vocab[len(vocab)] = new_token
    return

def find_frequent_pairs(token_arrays: list[int], vocab: dict[int, bytes], num_processes: int = 4, special_encoding: int = 256) -> list[tuple[int, int]]:
    # Count the frequency of adjacent byte pairs in the vocabulary
    pair_freq = {}
    for i in range(len(token_arrays) - 1):
        if token_arrays[i] == special_encoding or token_arrays[i + 1] == special_encoding:
            continue  # Skip pairs involving the special token
        pair = (token_arrays[i], token_arrays[i + 1])
        if pair in pair_freq:
            pair_freq[pair] += 1
        else:
            pair_freq[pair] = 1

    # Find the most frequent pair
    if not pair_freq:
        return []

    # Sort by frequency first (descending), then lexicographically (descending)
    sorted_pairs = sorted(pair_freq.items(), key=lambda x: (-x[1], -x[0][0], -x[0][1]))
    return [sorted_pairs[0][0]] if sorted_pairs else []

