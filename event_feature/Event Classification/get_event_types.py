import os
import re
import pandas as pd

def extract_event_types_from_excel(file_path):
    """
    Extracts and de-duplicates event types from the 'Name' field of an Excel file.

    Parameters:
    file_path: str - The full path to the Excel file

    Returns:
    tuple - (event_types_list, event_count)
    event_types_list: list - A list of unique event types (sorted)
    event_count: int - The total number of event types
    """

    try:
        df = pd.read_excel(file_path)
        print(f"File successfully read, data shape:: {df.shape}")
    except Exception as e:
        raise FileNotFoundError(f"Failed to read the file.: {str(e)}")

    # Check if the Name field exists.
    if 'Name' not in df.columns:
        raise ValueError(
            f"The field 'Name' was not found in the Excel file. The existing fields are:: {df.columns.tolist()}"
        )

    # Define the event type extraction function.
    def extract_event_name(name_str):
        """Extract the event type from the string in the Name field."""
        if pd.isna(name_str):  
            return None
        # Use regular expressions to match alphanumeric combinations at the beginning of the string.
        match = re.match(r'^([A-Za-z0-9]+)', str(name_str))
        return match.group(1) if match else None

    # Apply the extraction function and process the data.
    df['Event_Type'] = df['Name'].apply(extract_event_name)

    # Remove duplicates and null values.
    unique_event_types = df['Event_Type'].dropna().unique()

    # Sort and convert to a list.
    event_types_list = sorted([event for event in unique_event_types if event])

    return event_types_list


if __name__ == "__main__":
    unique_events = []
    dataset_name = ['attack incident'] # , 'high_value'
    platform = ['ARB', 'AVAX', 'Base', 'BSC', 'ETH', 'POL']
    for dn in dataset_name:
        for pf in platform:
            protocol_path = "../../dataset/" + dn + "/" + pf + "/"
            if os.path.exists(protocol_path):
                protocol_list = os.listdir(protocol_path)
                for pt in protocol_list:
                    EXCEL_FILE_PATH = protocol_path + pt + "/Event.xlsx"
                    events = extract_event_types_from_excel(EXCEL_FILE_PATH)
                    for e in events:
                        if e not in unique_events:
                            unique_events.append(e)

    print("\n" + "=" * 60)
    print("Extraction results of event types from Excel files")
    print("=" * 60)
    print(f"Total number of event types (unique): {len(unique_events)}")

    print("\nList of unique event types:")
    print("-" * 40)
    for index, event in enumerate(unique_events, 1):
        print(f"{index:2d}. {event}")

    print(f"\nPython list format results:")
    print("-" * 40)
    print(unique_events)

    with open("event_types_result.txt", "w", encoding="utf-8") as f:
        f.write(f"Total number of event types: {len(unique_events)}\n")
        f.write("List of event types:\n")
        for event in unique_events:
            f.write(f"- {event}\n")
    print(f"\nThe results have been saved to: event_types_result.txt")
