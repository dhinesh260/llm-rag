def merge_files(output_file, chunk_files):
    """
    Merge multiple chunk files into a single file.

    :param output_file: Path to the output file (reconstructed file)
    :param chunk_files: List of chunk file paths in the correct order
    """
    with open(output_file, "wb") as output:
        for chunk_file in chunk_files:
            print(f"Merging {chunk_file}...")
            with open(chunk_file, "rb") as chunk:
                output.write(chunk.read())
    print(f"Merge complete! File saved as: {output_file}")

# Example usage
if __name__ == "__main__":
    # List of chunk files in the correct order
    chunk_files = [
        "all-MiniLM-L6-v2.zip.part1",
        "all-MiniLM-L6-v2.zip.part2",
        "all-MiniLM-L6-v2.zip.part3",
        "all-MiniLM-L6-v2.zip.part4"
    ]
    
    # Path to the output file
    output_file = "reconstructed_example.zip"
    
    # Merge the files
    merge_files(output_file, chunk_files)
