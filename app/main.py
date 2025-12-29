from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import subprocess
from pathlib import Path
from typing import Optional
import logging
import shutil
import uuid

from app.downloader import download_video
from app.hls import convert_to_hls
from app.wasabi_uploader import WasabiUploader

logging.basicConfig(level=logging.INFO)
logger        = logging.getLogger(__name__)
app           = FastAPI()
DOWNLOAD_DIR  = "downloads"
HLS_DIR       = "hls_videos"
CLIP_DURATION = 3  # Default Clip Duration In Seconds

class VideoRequest(BaseModel):
    video_url:        str
    destination_path: str
    generate_clip:    bool = False

class VideoResponse(BaseModel):
    main_video_url:   str
    clip_url:         Optional[str] = None

def cleanup_files(*paths):
    for path in paths:
        try:
            if path and os.path.exists(path):
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

def create_clip(input_path: str, output_path: str, duration: int):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-t", str(duration),
        "-c", "copy",
        output_path
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )

    _, stderr = process.communicate()
    if process.returncode != 0:
        raise Exception(stderr)

    return output_path

@app.post("/convert-video/", response_model=VideoResponse)
async def convert_video(data: VideoRequest):
    temp_download_dir = None
    temp_hls_dir      = None
    downloaded_path   = None
    clip_path         = None

    try:
        if not data.destination_path or "/" not in data.destination_path:
            raise ValueError("Invalid Destination Path")

        uid = str(uuid.uuid4())

        temp_download_dir = os.path.join(DOWNLOAD_DIR, uid)
        temp_hls_dir      = os.path.join(HLS_DIR, uid)

        os.makedirs(temp_download_dir, exist_ok=True)
        os.makedirs(temp_hls_dir, exist_ok=True)

        downloaded_path = download_video(data.video_url, temp_download_dir)
        video_name      = Path(downloaded_path).stem
        if data.generate_clip:
            clip_dir  = os.path.join(temp_download_dir, "clip")
            os.makedirs(clip_dir, exist_ok=True)
            clip_output = os.path.join(
                clip_dir,
                f"{video_name}_clip{Path(downloaded_path).suffix}"
            )
            clip_path = create_clip(
                downloaded_path,
                clip_output,
                CLIP_DURATION
            )

        hls_main_dir = os.path.join(temp_hls_dir, "main")
        convert_to_hls(downloaded_path, hls_main_dir)

        hls_clip_dir = None
        if clip_path:
            hls_clip_dir = os.path.join(temp_hls_dir, "clip")
            convert_to_hls(clip_path, hls_clip_dir)

        uploader = WasabiUploader()

        uploader.upload_directory(
            local_path=hls_main_dir,
            destination_path=data.destination_path
        )

        if hls_clip_dir:
            uploader.upload_directory(
                local_path=hls_clip_dir,
                destination_path=f"{data.destination_path}/clip"
            )

        bucket    = data.destination_path.split("/")[0]
        base_path = "/".join(data.destination_path.split("/")[1:])
        main_url  = f"https://{bucket}.s3.us-west-1.wasabisys.com/{base_path}/playlist.m3u8"
        clip_url  = (
            f"https://{bucket}.s3.us-west-1.wasabisys.com/{base_path}/clip/playlist.m3u8"
            if hls_clip_dir else None
        )

        cleanup_files(temp_download_dir, temp_hls_dir)

        return VideoResponse(
            main_video_url=main_url,
            clip_url=clip_url
        )

    except Exception as e:
        logger.error(str(e))
        raise HTTPException(
            status_code=400 if isinstance(e, ValueError) else 500,
            detail=str(e)
        )

    finally:
        cleanup_files(temp_download_dir, temp_hls_dir)