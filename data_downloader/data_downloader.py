#data_downloader
from pathlib import Path
import os
import http.client
from tqdm import tqdm

def is_safe_path(basedir, path, follow_symlinks=True):  # Corrected here
    if follow_symlinks:
        return os.path.realpath(str(path)).startswith(str(basedir))
    return os.path.abspath(str(path)).startswith(str(basedir))

def download_file(url, filename):
    if not is_safe_path(Path.cwd(), filename):
        raise ValueError("Invalid filename")

    filename.parent.mkdir(parents=True, exist_ok=True)

    conn = http.client.HTTPSConnection("datashare.ed.ac.uk")
    conn.request("GET", "/bitstream/handle/10283/2353/SiwisFrenchSpeechSynthesisDatabase.zip?sequence=3&isAllowed=y")

    r = conn.getresponse()

    # Get the total file size from the headers
    file_size = int(r.getheader('Content-Length'))

    # Check if the file already exists and is the expected size
    if filename.exists() and filename.stat().st_size == file_size:
        print('File already exists and is complete, not downloading.')
        return
        
    # Create a progress bar
    progress = tqdm(total=file_size, ncols=100, unit='B', unit_scale=True)

    with open(filename, 'wb') as f:
        for chunk in iter(lambda: r.read(8192), b''):
            f.write(chunk)
            progress.update(len(chunk))

    # Close the progress bar
    progress.close()

# Usage
url = 'https://datashare.ed.ac.uk/bitstream/handle/10283/2353/SiwisFrenchSpeechSynthesisDatabase.zip?sequence=3&isAllowed=y'
filename = Path('dataset/SiwisFrenchSpeechSynthesisDatabase.zip')
try:
    download_file(url, filename)
except Exception as e:
    print(e)