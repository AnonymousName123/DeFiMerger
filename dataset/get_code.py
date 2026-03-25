import requests
import json
import os
import shutil
from typing import Optional, Dict, List
import pandas as pd
import time

final_results = []

def get_contract_source_code(
        contract_address: str,
        api_key: str,
        chainid: int = 1,
        number: int = 0,
        output_dir: str = "./contracts",
        is_proxy_recursion: bool = False,
        original_address: str = None
) -> bool:
    url = "https://api.etherscan.io/v2/api"
    params = {
        "module": "contract",
        "action": "getsourcecode",
        "address": contract_address,
        "apikey": api_key,
        "chainid": chainid,
    }

    if not original_address:
        original_address = contract_address

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "1":
            return False

        result_list = data.get("result", [])
        if not result_list:
            return False

        contract_data = result_list[0]
        implementation_address = contract_data.get("Implementation", "").strip()

        # Extract the source code and ABI information for the current layer.
        source_code = contract_data.get("SourceCode", "")
        contract_name = contract_data.get("ContractName", "")
        abi = contract_data.get("ABI", "")

        if implementation_address and implementation_address.lower() != contract_address.lower():
            # Attempting to retrieve the logic contract.
            success = get_contract_source_code(
                contract_address=implementation_address,
                api_key=api_key,
                chainid=chainid,
                number=number,
                output_dir=output_dir,
                is_proxy_recursion=True,
                original_address=original_address
            )
            # If the logic contract is successfully retrieved, return True.
            if success:
                return True
            else:
                print(f"[Fallback] Failed to retrieve the logic contract at {implementation_address}, attempting to save the proxy contract at {contract_address}.")
                # If the logic contract fails, do not exit, but continue executing the following "save current contract" logic.

        # Check if there is source code available at the current level.
        if not source_code or not contract_name or contract_name == "":
            return False

        # Directory preparation
        os.makedirs(f"{output_dir}/abi/", exist_ok=True)
        os.makedirs(f"{output_dir}/source/", exist_ok=True)

        # Save ABI
        with open(f"{output_dir}/abi/{number}_abi.json", "w", encoding="utf-8") as f:
            json.dump(json.loads(abi) if abi.startswith("[") else abi, f, indent=2)

        # Save the source code logic.
        success_save = False
        if source_code.startswith("{"):
            try:
                json_str = source_code.strip()
                if json_str.startswith("{{") and json_str.endswith("}}"):
                    json_str = json_str[1:-1]
                source_dict = json.loads(json_str)
                if "sources" in source_dict:
                    source_dict = source_dict["sources"]

                for filepath, content in source_dict.items():
                    filename = filepath.split('/')[-1]
                    # If there are multiple files, match the contract name; if there is only one file, save it directly.
                    if contract_name in filename or len(source_dict) == 1:
                        save_path = f"{output_dir}/source/{number}_{contract_name}.sol"
                        with open(save_path, "w", encoding="utf-8") as f:
                            f.write(content["content"])
                        success_save = True
            except Exception as e:
                print(f"Failed to parse JSON source code.: {e}")
                return False
        else:
            save_path = f"{output_dir}/source/{number}_{contract_name}.sol"
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(source_code)
            success_save = True

        if success_save:
            # Indicates whether the proxy contract was saved as a rollback version.
            tag = "[Proxy-Only]" if is_proxy_recursion == False and implementation_address else "[Success]"
            addr_str = f"{original_address} (Proxy Source)" if tag == "[Proxy-Only]" else f"{original_address} -> {contract_address}"
            log_msg = f"{tag} No.{number} | Name: {contract_name} | Addr: {addr_str}"
            print(log_msg)
            final_results.append(log_msg)
            return True

    except Exception as e:
        print(f"Error in get_contract_source_code: {e}")
        return False

    return False


if __name__ == "__main__":
    dataset = 'attack incident'
    platform = ['ARB', 'AVAX', 'Base', 'BSC', 'ETH', 'POL']
    chain_id_map = ['42161', '43114', '8453', '56', '1', '137']
    address_cache = {}
    API_KEYS = []
    count = 0

    for pt in platform:
        protocol_list = os.listdir(f'./{dataset}/{pt}/')
        current_chain_id = chain_id_map[platform.index(pt)]

        for protocol in protocol_list:
            protocol_path = f'./{dataset}/{pt}/{protocol}/'
            if not os.path.isdir(protocol_path): continue

            excel_path = f'{protocol_path}Event.xlsx'
            if not os.path.exists(excel_path): continue

            df = pd.read_excel(excel_path, sheet_name=0)


            def is_valid_number(x):
                try:
                    float(str(x))
                    return True
                except (ValueError, TypeError):
                    return False


            invalid_mask = df.iloc[:, 0].apply(is_valid_number)
            if not invalid_mask.all():
                print(f"Cleaning {protocol} excel rows...")
                clean_df = df[invalid_mask].copy()
                clean_df.iloc[:, 0] = clean_df.iloc[:, 0].apply(lambda x: int(float(str(x))))
                clean_df.to_excel(excel_path, index=False)
                df = clean_df
            else:
                df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: int(float(str(x))))

            events = df.values
            print(f'\n--- Start: {protocol} (Total: {len(events)}) ---')

            for event in events:
                FLAG = False # Determine if the contract exists.
                number = event[0]
                contract_addr = str(event[1]).lower().strip()[:42]

                contract_dir = os.listdir(f'{protocol_path}source/')
                for contract in contract_dir:
                    if contract.startswith(f'{number}_'):
                        FLAG = True
                if FLAG == True:
                    # print('skip ' + str(number))
                    address_cache[(current_chain_id, contract_addr)] = (number, protocol_path)
                    continue

                cache_key = (current_chain_id, contract_addr)
                if cache_key in address_cache:
                    old_num, old_path = address_cache[cache_key]
                    try:
                        for folder in ['source', 'abi', 'meta']:
                            src_dir, dst_dir = f"{old_path}{folder}/", f"{protocol_path}{folder}/"
                            os.makedirs(dst_dir, exist_ok=True)
                            for f_name in os.listdir(src_dir):
                                if f_name.startswith(f"{old_num}_"):
                                    shutil.copy2(os.path.join(src_dir, f_name),
                                                 os.path.join(dst_dir, f_name.replace(f"{old_num}_", f"{number}_", 1)))
                        # Record cache hit occurrences.
                        cache_msg = f"[Cache Hit] No.{number} | Addr: {contract_addr} (Copied from No.{old_num})"
                        print(cache_msg)
                        final_results.append(cache_msg)
                        continue
                    except:
                        pass

                time.sleep(0.3)
                success = get_contract_source_code(
                    contract_address=contract_addr,
                    api_key=API_KEYS[count],
                    chainid=int(current_chain_id),
                    number=number,
                    output_dir=protocol_path
                )

                count = (count + 1) % len(API_KEYS)

                if success:
                    address_cache[cache_key] = (number, protocol_path)

    with open("contract_mapping.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(final_results))
    print(f"\n[Finished] All records have been saved to contract_mapping.txt")