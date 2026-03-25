import openpyxl

def extract_data_with_multi_tx(file_path):

    wb = openpyxl.load_workbook(file_path, data_only=True)
    sheet = wb.active

    url_list = []
    protocol_names = [] 

    # Define the browser domain mapping corresponding to the chain.
    chain_explorer_map = {
        'ETH': 'https://etherscan.io',
        'BSC': 'https://bscscan.com',
        'BASE': 'https://basescan.org',
        'POL': 'https://polygonscan.com',
        'ARB': 'https://arbiscan.io',
        'FTM': 'https://ftmscan.com/',
        'AVAX': 'https://snowtrace.io'
    }

    print("Starting to process the data....")

    # Iterate through each row (skip the header row, starting from the second row)
    for row in sheet.iter_rows(min_row=2, values_only=False):

        p_name = row[1].value  # Protocol name
        c_name = row[2].value  # Chain name
        tx_raw = row[5].value  # Original tx data

        if not p_name or not c_name or not tx_raw:
            continue

        protocol_str = str(p_name).strip()
        chain_str = str(c_name).strip().upper()

        tx_hashes = str(tx_raw).split()

        base_url = chain_explorer_map.get(chain_str)

        if base_url:
            for tx in tx_hashes:
                # The length is greater than 10, to prevent corrupted data.
                if len(tx) > 10:
                    # Generate the URL and save it.
                    full_url = f"{base_url}/tx/{tx}#eventlog"
                    url_list.append(full_url)

                    # Also store the corresponding protocol name (maintaining consistent indexing).
                    protocol_names.append(protocol_str)
        else:
            print(f"Note: No corresponding URL was found for chain '{chain_str}', and protocol '{protocol_str}' has been skipped.")

    return protocol_names, url_list


file_name = 'dataset.xlsx' 

try:
    protocols, urls = extract_data_with_multi_tx(file_name)

    print(f"\nSuccessfully extracted {len(urls)} data entries (including multiple transaction splits):")
    print("-" * 50)
    print(f"{'Protocol':<20} | {'URL'}")
    print("-" * 50)

    for p, u in zip(protocols, urls):
        print(f"{p:<20} | {u}")

except FileNotFoundError:
    print(f"Error: File '{file_name}' not found. Please ensure you are running this code locally and that the original .xlsx file exists.")