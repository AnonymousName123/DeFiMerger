import requests
from bs4 import BeautifulSoup
import os

def process(filepath):
    file = open(filepath, 'r', encoding='utf-8')
    text = file.readlines()
    file.close()

    flag = False
    new_text = ''
    num = 0
    while num < len(text):
        line = text[num]
        # print(line)
        if line == "(Showing the first 1000 log entries only)\n":
            num += 1
            continue

        if flag == True:
            if line == 'Name\n':
                new_text += line
                name = ''
                num += 1
                while text[num] != 'View Source\n':
                    name += text[num][:-1] + ' '
                    num += 1
                new_text += name + '\n'

            elif line == 'Data\n':
                new_text += line
                while text[num].startswith('Authority') == False and text[num].startswith('Txn') == False and text[num] != 'Address\n':
                    if ':' in text[num]:
                        new_text += text[num][:-1] + ' '
                        new_text += text[num+1] + '\n'
                    num += 1
                if text[num] == 'Address\n':
                    new_text += text[num-1]
                    new_text += text[num]
                else:
                    flag = False
                    break

            else:
                new_text += line

        if line == "Transaction Receipt Event Logs\n":
            flag = True

        num += 1
    return new_text


def save_url_content_to_txt(urls, output_dir="url_contents"):
    """
    Save the text content of the given URL to a local TXT file.
    
    Parameter:
        urls (list): List containing URLs
        output_dir (str): Output directory, defaults to "url_contents"
    """
    # Create the output directory (if it doesn't exist).
    filepaths = []
    os.makedirs(output_dir, exist_ok=True)
    
    for url in urls:
        try:
            # Get website content
            headers = {
                'User-Agent': '' # Write your User-Agent
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()  # Check if the request was successful.
            
            # Parse HTML and extract text.
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(separator='\n', strip=True)
            
            # Generate filename (extracted from the URL)
            filename = url.replace("https://", "").replace("http://", "").replace("/", "_")
            if len(filename) > 100:  # Preventing excessively long filenames
                filename = filename[:100]
            filepath = os.path.join(output_dir, f"{filename}.txt")
            filepaths.append(filepath)
            
            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"URL: {url}\n\n")
                f.write(text)
            
            print(f"Successfully saved: {filepath}")
            
        except Exception as e:
            print(f"An error occurred while processing the URL {url}: {str(e)}")

    return filepaths


if __name__ == "__main__":
    # Replace this with the list of URLs you want to scrape.
    urls = []
    
    filepaths = save_url_content_to_txt(urls)
    for filepath in filepaths:
        text = process(filepath)

        file = open(filepath, 'w', encoding='utf-8')
        file.write(text)
        file.close()