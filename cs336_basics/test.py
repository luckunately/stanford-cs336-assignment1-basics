import pretokenizer

def test_find_chunk_boundaries():
    test_data = b"Hello<|endoftext|>World<|endoftext|>This<|endoftext|>Is<|endoftext|>A<|endoftext|>Test"
    split_token = b"<|endoftext|>"
    desired_chunks = 5

    with open("test_file.txt", "wb") as f:
        f.write(test_data)

    with open("test_file.txt", "rb") as f:
        chunk_boundaries = pretokenizer.find_chunk_boundaries(f, desired_chunks, split_token)

    print("chunks:", chunk_boundaries)
    # print the content of each chunk
    with open("test_file.txt", "rb") as f:
        for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start)
            print(f"Chunk ({start}-{end}):", chunk)
    
    # test for regex_tokenize
    tokens = pretokenizer.regex_tokenize(test_data, split_token)
    reverted = pretokenizer.revert_regex_tokenize(tokens)
    print("Original:", test_data)
    print("Tokens:", tokens)
    print("Reverted:", reverted)
    assert reverted == test_data, "Reverted tokens do not match original data"
    
if __name__ == "__main__":
    test_find_chunk_boundaries()