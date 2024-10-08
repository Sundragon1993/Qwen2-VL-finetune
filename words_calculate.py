import os
from collections import Counter

# Function to count the frequency of words "right" and "left" in a text file
def count_words_in_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read().lower()  # Read the file content and convert to lowercase for case-insensitive matching
        words = content.split()
        counter = Counter(words)
        return counter['right'], counter['left']

# Function to traverse through the folder and calculate the total occurrences
def calculate_word_frequencies(folder_path):
    right_count = 0
    left_count = 0

    for subdir, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):  # Only process .txt files
                file_path = os.path.join(subdir, file)
                right, left = count_words_in_file(file_path)
                right_count += right
                left_count += left

    return right_count, left_count

# Example usage
folder_path = '/home/admin/finetune-Qwen2-VL/VLM'  # Replace with the path to your folder
right_count, left_count = calculate_word_frequencies(folder_path)
print(f"Total 'right' count: {right_count}")
print(f"Total 'left' count: {left_count}")
