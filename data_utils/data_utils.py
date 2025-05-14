def pad_sequence(sequence: list[int], pad_token: int, max_length: int) -> list[int]:
    for _ in range(max_length - len(sequence)):
        sequence.append(pad_token)

    return sequence

def pad_sequences(sequences: list[list[int]], pad_token: int, max_length: int) -> list[list[int]]:
    return [pad_sequence(sequence, pad_token, max_length) for sequence in sequences]