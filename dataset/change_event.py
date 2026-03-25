import re
import os
import shutil
from openpyxl import Workbook

data_name = 'attack incident' # high_value, attack incident
platform = ['ARB', 'AVAX', 'Base', 'BSC', 'ETH', 'POL']
def parse_eth_events(text):
    # Split by event (using lines starting with a number as separators)
    events = []
    current_event = {}
    lines = text.split('\n')
    count = 0
    
    for num in range(len(lines)):
        line = lines[num].strip()
        if not line:
            continue
            
        # Matching Number
        if line == 'Address':
            if current_event:
                events.append(current_event)
                current_event = {}
            current_event['Number'] = lines[num-1]
            current_event['Address'] = lines[num+1]
            
        # Matching Name
        elif line == 'Name':
            name = lines[num+1].replace('View Source', '').strip() # attack incidents

            """
            # high_value
            pos = 1
            name = lines[num+pos] + ' '
            pos += 1
            while lines[num+pos] != 'View Source':
                name += lines[num+pos] + ' '
                pos += 1
            """

            count = name.count('index_topic')
            current_event['Name'] = name
            
        # Matching Topics
        elif line == 'Topics':
            pos = 1
            topics_data = []
            while lines[num+pos] != 'Data':
                if lines[num+pos].startswith('0x') and lines[num+pos-1] != '0':
                    topics_data.append(lines[num+pos])
                elif re.match(r'^\d+$', lines[num+pos]) and ':' not in lines[num+pos] and '0x' not in lines[num+pos]:
                    topics_data.append(lines[num+pos])
                elif "channelId" in lines[num+pos-1]:
                    topics_data.append(lines[num + pos])
                pos += 1

            # Fixing topics errors
            if count != len(topics_data):
                i = 0
                count = 0
                while i < len(topics_data):
                    del topics_data[i]
                    i += 1

            current_event['Topics'] = topics_data
            
        # Matching Data
        elif line == 'Data':
            pos = 1
            data = []
            while num + pos + 1 < len(lines) and lines[num+pos+1] != 'Address':
                if re.match(r'^\d+$', lines[num+pos]) or ':' in lines[num+pos] or lines[num+pos].startswith('0x'):
                    index = lines[num+pos].find(':')
                    data.append(lines[num+pos][index+1:].strip())
                pos += 1

            # Complete the last line.
            if num + pos + 1 == len(lines) and ':' in lines[-1]:
                index = lines[-1].find(':')
                data.append(lines[-1][index+1:].strip())

            current_event['Data'] = data

    
    if current_event:
        events.append(current_event)
    
    return events

def save_to_excel(events, output_file):
    wb = Workbook()
    ws = wb.active
    ws.title = 'events'
    
    ws.append(['Number', 'Address', 'Name', 'Topics', 'Data'])
    
    # Write data
    for event in events:
        number = event.get('Number', '')
        address = event.get('Address', '')
        name = event.get('Name', '')
        
        # Merge Topics (skip item 0)
        topics = '\n'.join(event.get('Topics', []))
        
        # Merge data (which may have multiple fields)
        data = '\n'.join(event.get('Data', []))
        
        ws.append([number, address, name, topics, data])
    
    wb.save(output_file)
    print(f"The Excel file has been saved.: {output_file}")

# Parse and save to Excel
for pf in platform:

    tmp = os.listdir('./' + data_name + '/' + pf)

    files = []
    folders = []
    for i in tmp:
        if 'txt' in i:
            files.append(i)
        else:
            folders.append(i)

    for i in folders:
        if os.path.exists('./' + data_name + '/' + pf + '/' + i + '/Event.xlsx'):
            continue
        else:
            print(i)
            file = open('./' + data_name + '/' + pf + '/' + i + '/event.txt', 'r', encoding='utf-8')
            event_data = file.read()
            file.close()

            events = parse_eth_events(event_data)
            output_file = './' + data_name + '/' + pf + '/' + i + '/Event.xlsx'
            save_to_excel(events, output_file)

    """
    for i in files:
        file = open('./' + data_name + '/' + pf + '/' + i, 'r', encoding='utf-8')
        event_data = file.read()
        file.close()

        os.makedirs('./' + data_name + '/' + pf + '/' + i[19:-13], exist_ok=True)
        events = parse_eth_events(event_data)
        output_file = './' + data_name + '/' + pf + '/' + i[19:-13] + '/Event.xlsx'
        save_to_excel(events, output_file)

        source_path = './' + data_name + '/' + pf + '/' + i
        dest_path = './' + data_name + '/' + pf + '/' + i[19:-13] + '/event.txt'
        shutil.copyfile(source_path, dest_path)
        os.remove('./' + data_name + '/' + pf + '/' + i)
    """

