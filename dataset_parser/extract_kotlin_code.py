import os
import shutil
import argparse


def extract_kotlin_files(source_dir, destination_dir):
    """
    Recursively extracts all Kotlin files from the source directory and copies them
    into a new subdirectory within the source directory.

    Args:
    source_dir (str): The path to the source directory where the Kotlin files are located.
    destination_dir (str): The path to the destination directory where the Kotlin files will be copied.
    """
    # Ensure the destination directory exists
    os.makedirs(destination_dir, exist_ok=True)

    # Walk through the source directory
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.kt'):
                source_file = os.path.join(root, file)
                shutil.copy(source_file, destination_dir)
                print(f"Copied: {source_file} to {destination_dir}")


def main():
    parser = argparse.ArgumentParser(description='Extract Kotlin files from a directory to a specified subdirectory.')
    parser.add_argument('source_dir', type=str, help='The source directory to search for Kotlin files.')
    parser.add_argument('new_folder', type=str, help='The name of the new folder to store extracted Kotlin files.')

    args = parser.parse_args()

    destination_directory = os.path.join(args.source_dir, args.new_folder)
    extract_kotlin_files(args.source_dir, destination_directory)


if __name__ == "__main__":
    main()
