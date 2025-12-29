import os
import time
import requests
from pathlib import Path
from urllib.parse import urlparse, unquote

def get_filename_from_url(url: str) -> str:
    parsed   = urlparse(url)
    filename = unquote(Path(parsed.path).name)
    return filename or f"video_{int(time.time())}.mp4"

def download_video(
    video_url:  str,
    save_dir:   str,
    timeout:    int = 30,
    chunk_size: int = 1024 * 1024
) -> str:

    if not video_url or not isinstance(video_url, str):
        raise ValueError("Invalid URL")

    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    if not save_dir_path.is_dir():
        raise ValueError(f"Invalid directory: {save_dir}")

    filename  = get_filename_from_url(video_url)
    file_path = save_dir_path / filename
    temp_path = f"{file_path}.download"

    with requests.Session() as session:
        session.head(video_url, timeout=timeout).raise_for_status()

        with session.get(video_url, stream=True, timeout=timeout) as response:
            response.raise_for_status()

            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)

    if os.path.exists(file_path):
        os.remove(file_path)

    os.rename(temp_path, file_path)
    return str(file_path)