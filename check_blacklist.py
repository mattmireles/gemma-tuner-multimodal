import pandas as pd
import glob
import os

def find_latest_blacklist_file(directory):
  """Finds the latest file starting with 'blacklist-' in the given directory.

  Args:
    directory: The directory to search in.

  Returns:
    The path to the latest 'blacklist-' file, or None if no such file is found.
  """
  list_of_files = glob.glob(os.path.join(directory, 'blacklist-*.csv'))
  if not list_of_files:
    return None
  latest_file = max(list_of_files, key=os.path.getmtime)
  return latest_file

def print_matching_lines(file1, file2):
  """Prints lines from file1 where the ID is also present in file2.

  Args:
    file1: Path to the first CSV file.
    file2: Path to the second CSV file.
  """
  try:
    df1 = pd.read_csv(file1, dtype={'id': str})  # Read first file, treating 'id' as string
    df2 = pd.read_csv(file2, usecols=['id'], dtype={'id': str}) # Read only 'id' from the second file

    # Remove empty IDs from df2 for matching
    valid_ids_df2 = df2['id'].dropna()

    # Merge based on 'id' to get matching rows
    merged_df = pd.merge(df1, valid_ids_df2, on='id', how='inner')

    if not merged_df.empty:
      print(f"Lines from {file1} with matching IDs in {file2}:")
      print(merged_df.to_string(index=False))  # Print the matching lines without index
    else:
      print("No matching IDs found.")

  except FileNotFoundError:
    print(f"Error: One or both of the files were not found.")
  except pd.errors.EmptyDataError:
    print("Error: One or both of the CSV files are empty.")
  except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Example usage:
file1 = find_latest_blacklist_file('output')

files = [
'data_patches/data3/delete/data3_prepared - remove - translated.csv',
'data_patches/data3/do_not_blacklist/blacklist - Keep ground-truth.csv',
'data_patches/data3/override_text_perfect/blacklist - Keep Edited.csv',
'data_patches/data3/override_text_perfect/data3_prepared - edited.csv',
]

"""
data_patches/
data_patches/data3
data_patches/data3/delete
data_patches/data3/do_not_blacklist
data_patches/data3/override_text_perfect
data_patches/data3/override_text_perfect/blacklist - Keep Edited.csv XXXX
"""

for x in files:
    print_matching_lines(file1, x)


















