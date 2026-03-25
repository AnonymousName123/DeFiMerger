import os
import pandas as pd
import re


def rename_folders_by_excel(target_directory, excel_path):
    # Read XLSX files
    try:
        df = pd.read_excel(excel_path)

        # Remove spaces from column names.
        df.columns = [c.strip() for c in df.columns]

        # Verify the necessary columns.
        required_cols = ['Transaction Hash', 'Protocol']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: The Excel file must contain the following columns.: {required_cols}")
            return
    except Exception as e:
        print(f"Failed to read the Excel file.: {e}")
        return

    # Get all subfolders under the target folder.
    if not os.path.exists(target_directory):
        print(f"Error: Path not found. -> {target_directory}")
        return

    folders = [f for f in os.listdir(target_directory) if os.path.isdir(os.path.join(target_directory, f))]
    print(f"Found {len(folders)} folders to process.\n")

    renamed_count = 0

    # Matching and renaming logic
    for folder_name in folders:
        match_found = False
        new_name = None

        # Iterate through each row of the Excel file.
        for _, row in df.iterrows():
            # Ensure that the hash is converted to a string to prevent errors caused by purely numerical hashes.
            tx_hash_val = str(row['Transaction Hash'])
            protocol_val = str(row['Protocol'])

            # Determine whether the folder name is a subsequence (contains) of the Transaction Hash.
            if folder_name.lower() in tx_hash_val.lower() and folder_name.strip() != "":
                match_found = True
                new_name = protocol_val
                break

        if match_found and new_name:
            # Remove illegal characters (characters not allowed in filenames).
            safe_new_name = re.sub(r'[\\/*?:"<>|]', "", new_name).strip()

            old_path = os.path.join(target_directory, folder_name)
            new_path = os.path.join(target_directory, safe_new_name)

            # Resolving name conflicts
            counter = 2
            original_new_path = new_path
            while os.path.exists(new_path):
                if old_path == new_path:
                    break
                new_path = f"{original_new_path}({counter})"
                counter += 1

            # Perform the renaming.
            if old_path != new_path:
                try:
                    os.rename(old_path, new_path)
                    print(f"[Success] {folder_name} -> {os.path.basename(new_path)}")
                    renamed_count += 1
                except Exception as e:
                    print(f"[Fail] Unable to rename {folder_name}: {e}")
        else:
            print(f"[Not matched] {folder_name}")

    print(f"\nTask completed! A total of {renamed_count} folders were successfully renamed.")


my_folders_path = './attack incident/POL/'

my_excel_file = "dataset.xlsx"

if __name__ == "__main__":
    rename_folders_by_excel(my_folders_path, my_excel_file)