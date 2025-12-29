import os
import shutil
import subprocess
import multiprocessing
from multiprocessing import Pool
import logging
import json
import math
import time
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# -------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "hls_converter.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("HLSConverter")

# -------------------------------------------------------------------
# Exceptions
# -------------------------------------------------------------------
class HLSConverterError(Exception):
    pass

# -------------------------------------------------------------------
# Caching
# -------------------------------------------------------------------
_MIG_CACHE = None
_GPU_DEVICES_CACHE = None

def get_gpu_info() -> str:
    """Get detailed GPU information"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=gpu_name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2
        )
        return result.stdout.strip()
    except Exception:
        return "Unknown GPU"

def get_available_gpu_devices() -> List[int]:
    """Get list of available GPU/MIG device indices (cached)"""
    global _GPU_DEVICES_CACHE
    if _GPU_DEVICES_CACHE is not None:
        return _GPU_DEVICES_CACHE

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2
        )
        devices = [int(idx.strip()) for idx in result.stdout.strip().split('\n') if idx.strip()]
        _GPU_DEVICES_CACHE = devices if devices else [0]

        # Log GPU info
        gpu_info = get_gpu_info()
        logger.info(f"Detected GPU devices: {_GPU_DEVICES_CACHE}")
        logger.info(f"GPU Info: {gpu_info}")

        return _GPU_DEVICES_CACHE
    except Exception as e:
        logger.warning(f"Could not detect GPU devices, defaulting to device 0: {e}")
        _GPU_DEVICES_CACHE = [0]
        return _GPU_DEVICES_CACHE

def check_mig_enabled() -> bool:
    """Check if MIG mode is enabled (cached)"""
    global _MIG_CACHE
    if _MIG_CACHE is not None:
        return _MIG_CACHE

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=mig.mode.current", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2
        )
        mig_status = result.stdout.strip().lower()
        _MIG_CACHE = "enabled" in mig_status
        return _MIG_CACHE
    except Exception:
        _MIG_CACHE = False
        return False

# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------
def _log_timing_summary(timings: Dict[str, float], file_size_mb: float, duration: float, mode: str = "Single"):
    """Log detailed timing summary"""
    logger.info("=" * 70)
    logger.info(f"📊 PERFORMANCE SUMMARY ({mode} Processing)")
    logger.info("=" * 70)
    logger.info(f"Input file size: {file_size_mb:.2f} MB")
    logger.info(f"Video duration: {duration:.2f}s")
    logger.info("")
    logger.info("⏱️  Stage-by-stage timing:")

    for stage, elapsed in timings.items():
        if stage != 'total':
            percentage = (elapsed / timings.get('total', elapsed)) * 100
            logger.info(f"  • {stage.replace('_', ' ').title()}: {elapsed:.2f}s ({percentage:.1f}%)")

    logger.info("")
    logger.info(f"⏱️  Total time: {timings.get('total', 0):.2f}s")

    if duration > 0:
        speed_ratio = duration / timings.get('total', 1)
        logger.info(f"🚀 Speed: {speed_ratio:.2f}x realtime")
        logger.info(f"📈 Throughput: {file_size_mb / timings.get('total', 1):.2f} MB/s")

    logger.info("=" * 70)

def get_video_info(input_path: str) -> Dict[str, float]:
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,duration,r_frame_rate,codec_name",
            "-of", "json",
            input_path
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )

        data = json.loads(result.stdout)

        if not data.get("streams"):
            raise HLSConverterError("No video stream found")

        stream = data["streams"][0]
        num, den = map(float, stream.get("r_frame_rate", "30/1").split("/"))

        return {
            "width": int(stream.get("width", 0)),
            "height": int(stream.get("height", 0)),
            "duration": float(stream.get("duration", 0)),
            "fps": num / den if den else 30,
            "codec": stream.get("codec_name", "h264")
        }

    except Exception as e:
        logger.exception("Failed to fetch video info")
        raise HLSConverterError(str(e))

# -------------------------------------------------------------------
# Segment Processing (for parallel encoding)
# -------------------------------------------------------------------
def _process_video_segment(args: Tuple) -> Tuple[bool, str]:
    """Process a single video segment - used for parallel processing"""
    (segment_idx, input_path, output_path, start_time, duration,
     video_bitrate, audio_bitrate, crf, gpu_device, mig_enabled, cpu_threads) = args

    try:
        cmd = ["ffmpeg", "-y"]

        if mig_enabled:
            # MIG mode: GPU-accelerated decoding with CPU encoding
            cmd += [
                "-hwaccel", "cuda",
                "-hwaccel_device", str(gpu_device),
                "-extra_hw_frames", "8",
                "-ss", str(start_time),
                "-t", str(duration),
                "-i", input_path,
                "-threads", str(cpu_threads),
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", str(crf),
                "-maxrate", video_bitrate,
                "-bufsize", str(int(video_bitrate[:-1]) * 2) + "k",
                "-pix_fmt", "yuv420p",
                "-profile:v", "main",
                "-x264-params", f"threads={cpu_threads}:lookahead-threads=2",
            ]
        else:
            # NVENC mode
            cmd += [
                "-hwaccel", "cuda",
                "-hwaccel_output_format", "cuda",
                "-hwaccel_device", str(gpu_device),
                "-extra_hw_frames", "8",
                "-ss", str(start_time),
                "-t", str(duration),
                "-i", input_path,
                "-c:v", "h264_nvenc",
                "-preset", "p2",
                "-tune", "ll",
                "-rc", "vbr",
                "-b:v", video_bitrate,
                "-maxrate", video_bitrate,
                "-bufsize", str(int(video_bitrate[:-1]) * 2) + "k",
                "-profile:v", "main",
                "-spatial_aq", "1",
                "-temporal_aq", "1",
            ]

        # Audio settings
        cmd += [
            "-c:a", "aac",
            "-b:a", audio_bitrate,
            "-ac", "2",
            "-ar", "48000",
            "-f", "mpegts",
            output_path
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )

        logger.info(f"Segment {segment_idx} processed successfully on GPU {gpu_device}")
        return (True, output_path)

    except Exception as e:
        error_msg = f"Segment {segment_idx} failed: {str(e)}"
        logger.error(error_msg)
        return (False, error_msg)

def _generate_hls_parallel(
    input_path: str,
    output_dir: Path,
    segment_dir: Path,
    info: Dict,
    video_bitrate: str,
    audio_bitrate: str,
    hls_time: int,
    crf: int,
    gpu_device: int,
    mig_enabled: bool,
    parallel_workers: Optional[int]
) -> str:
    """Generate HLS using parallel segment processing"""

    duration = info["duration"]

    # Determine number of workers
    cpu_count = multiprocessing.cpu_count()
    if parallel_workers is None:
        # Use 4 workers by default for parallel processing
        num_workers = min(4, cpu_count // 2)
    else:
        num_workers = min(parallel_workers, cpu_count)

    logger.info(f"Using {num_workers} parallel workers for encoding")

    # Split video into chunks for parallel processing
    # Each chunk should be at least 30 seconds
    chunk_duration = max(30, duration / num_workers)
    num_chunks = math.ceil(duration / chunk_duration)

    # Calculate threads per worker
    threads_per_worker = max(2, cpu_count // num_workers)

    temp_dir = output_dir / "temp_segments"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Prepare arguments for each segment
    segment_args = []
    for i in range(num_chunks):
        start_time = i * chunk_duration
        seg_duration = min(chunk_duration, duration - start_time)
        output_path = str(temp_dir / f"chunk_{i:03d}.ts")

        segment_args.append((
            i, input_path, output_path, start_time, seg_duration,
            video_bitrate, audio_bitrate, crf, gpu_device, mig_enabled, threads_per_worker
        ))

    # Process segments in parallel
    logger.info(f"Processing {num_chunks} chunks in parallel...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process_video_segment, args) for args in segment_args]
        results = []

        for future in as_completed(futures):
            success, result = future.result()
            if not success:
                raise HLSConverterError(f"Parallel processing failed: {result}")
            results.append(result)

    # Concatenate all segments
    logger.info("Concatenating segments...")
    concat_file = temp_dir / "concat_list.txt"

    # Sort results by filename to maintain order
    results.sort()
    with open(concat_file, 'w') as f:
        for segment_path in results:
            f.write(f"file '{segment_path}'\n")

    # Create final HLS output
    final_temp = segment_dir / "temp_full.ts"
    concat_cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        str(final_temp)
    ]

    subprocess.run(concat_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Generate HLS from concatenated file
    playlist_path = segment_dir / "playlist.m3u8"

    hls_cmd = [
        "ffmpeg", "-y",
        "-i", str(final_temp),
        "-c", "copy",
        "-f", "hls",
        "-hls_time", str(hls_time),
        "-hls_playlist_type", "vod",
        "-hls_flags", "independent_segments+temp_file",
        "-hls_segment_filename", str(segment_dir / "segment_%03d.ts"),
        "-start_number", "0",
        str(playlist_path)
    ]

    subprocess.run(hls_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Clean up temp files
    shutil.rmtree(temp_dir, ignore_errors=True)
    final_temp.unlink(missing_ok=True)

    # Create master playlist
    master_playlist = output_dir / "playlist.m3u8"
    master_playlist.write_text(
        "#EXTM3U\n"
        "#EXT-X-VERSION:3\n"
        f"#EXT-X-STREAM-INF:BANDWIDTH=2000000,RESOLUTION={info['width']}x{info['height']}\n"
        "segment/playlist.m3u8\n"
    )

    logger.info(f"Parallel HLS conversion completed successfully")
    return str(master_playlist)

# -------------------------------------------------------------------
# Core Logic
# -------------------------------------------------------------------
def generate_hls(
    input_path: str,
    output_dir: str,
    video_bitrate: str = "2000k",
    audio_bitrate: str = "128k",
    hls_time: int = 2,
    crf: int = 23,
    delete_temp: bool = True,
    gpu_device: int = 0,
    enable_parallel: bool = False,
    parallel_workers: Optional[int] = None
) -> str:

    # Start total timing
    total_start = time.time()
    timings = {}

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise HLSConverterError("FFmpeg/FFprobe not installed")

    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)

    # Get file size
    file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    logger.info(f"Input file: {input_path} ({file_size_mb:.2f} MB)")

    # Check if MIG is enabled (cached)
    mig_enabled = check_mig_enabled()
    if mig_enabled:
        logger.warning("MIG mode is enabled - NVENC not available. Using GPU-accelerated decoding with CPU encoding.")

    output_dir = Path(output_dir)
    segment_dir = output_dir / "segment"
    cpu_count = multiprocessing.cpu_count()

    try:
        # Stage 1: Video info extraction
        probe_start = time.time()
        info = get_video_info(input_path)
        timings['video_probe'] = time.time() - probe_start
        logger.info(f"⏱️  Video probe completed: {timings['video_probe']:.2f}s")
        duration = info["duration"]
        logger.info(f"Video info: {info['width']}x{info['height']}, {duration:.2f}s, {info['fps']:.2f} fps, codec: {info['codec']}")

        # Stage 2: Directory setup
        setup_start = time.time()
        output_dir.mkdir(parents=True, exist_ok=True)
        segment_dir.mkdir(parents=True, exist_ok=True)
        timings['directory_setup'] = time.time() - setup_start

        fps = round(info["fps"])
        gop = fps * hls_time

        playlist_path = segment_dir / "playlist.m3u8"
        master_playlist = output_dir / "playlist.m3u8"

        # Parallel processing for longer videos
        if enable_parallel and duration > 30:
            logger.info(f"Using parallel processing for video (duration: {duration}s)")

            # Stage 3: Parallel encoding
            encode_start = time.time()
            result = _generate_hls_parallel(
                input_path, output_dir, segment_dir, info,
                video_bitrate, audio_bitrate, hls_time, crf,
                gpu_device, mig_enabled, parallel_workers
            )
            timings['encoding'] = time.time() - encode_start
            timings['total'] = time.time() - total_start

            _log_timing_summary(timings, file_size_mb, duration, "Parallel")
            return result

        # Stage 3: Single-process encoding
        encode_start = time.time()

        # Base command
        cmd = [
            "ffmpeg",
            "-y",
        ]

        if mig_enabled:
            # MIG mode: GPU-accelerated decoding with CPU encoding
            logger.info(f"Using CUDA-accelerated decoding with CPU encoding (MIG mode) on GPU {gpu_device}")
            cmd += [
                "-hwaccel", "cuda",
                "-hwaccel_device", str(gpu_device),
                "-extra_hw_frames", "8",
                "-i", input_path,
                "-threads", str(cpu_count),
                "-fps_mode", "cfr",
                "-r", str(fps),
                "-g", str(gop),
                "-keyint_min", str(gop),
                "-sc_threshold", "0",
                "-c:v", "libx264",
                "-preset", "veryfast",  # Changed from "faster" for better speed
                "-crf", str(crf),
                "-maxrate", video_bitrate,
                "-bufsize", str(int(video_bitrate[:-1]) * 2) + "k",
                "-pix_fmt", "yuv420p",
                "-profile:v", "main",
                "-movflags", "+faststart",
                "-x264-params", f"threads={cpu_count}:lookahead-threads=4",
            ]
        else:
            # Full GPU mode: NVENC encoding
            logger.info(f"Using NVENC hardware encoding (full GPU mode) on GPU {gpu_device}")
            cmd += [
                "-hwaccel", "cuda",
                "-hwaccel_output_format", "cuda",
                "-hwaccel_device", str(gpu_device),
                "-extra_hw_frames", "8",
                "-i", input_path,
                "-fps_mode", "cfr",
                "-r", str(fps),
                "-g", str(gop),
                "-keyint_min", str(gop),
                "-sc_threshold", "0",
                "-c:v", "h264_nvenc",
                "-preset", "p2",  # Changed from p4 to p2 for faster encoding
                "-tune", "ll",  # Changed from hq to ll (low latency) for speed
                "-rc", "vbr",
                "-b:v", video_bitrate,
                "-maxrate", video_bitrate,
                "-bufsize", str(int(video_bitrate[:-1]) * 2) + "k",
                "-profile:v", "main",
                "-spatial_aq", "1",
                "-temporal_aq", "1",
                "-2pass", "0",
            ]

        # Audio settings (optimized)
        cmd += [
            "-c:a", "aac",
            "-b:a", audio_bitrate,
            "-ac", "2",
            "-ar", "48000",
        ]

        # HLS settings (optimized)
        cmd += [
            "-f", "hls",
            "-hls_time", str(hls_time),
            "-hls_playlist_type", "vod",
            "-hls_flags", "independent_segments+temp_file",  # Added temp_file for atomic writes
            "-hls_segment_filename", str(segment_dir / "segment_%03d.ts"),
            "-start_number", "0",
            str(playlist_path)
        ]

        logger.info(f"Starting HLS conversion: {input_path}")

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )

        timings['encoding'] = time.time() - encode_start
        logger.info(f"⏱️  Encoding completed: {timings['encoding']:.2f}s")

        # Verify playlist was created
        if not playlist_path.exists():
            raise HLSConverterError("HLS generation failed - no playlist created")

        # Create master playlist
        master_playlist_content = (
            "#EXTM3U\n"
            "#EXT-X-VERSION:3\n"
            f"#EXT-X-STREAM-INF:BANDWIDTH=2000000,RESOLUTION={info['width']}x{info['height']}\n"
            "segment/playlist.m3u8\n"
        )

        master_playlist.write_text(master_playlist_content)

        timings['total'] = time.time() - total_start
        _log_timing_summary(timings, file_size_mb, duration, "Single")

        return str(master_playlist)

    except subprocess.CalledProcessError as e:
        stderr_output = e.stderr.decode('utf-8', errors='ignore')
        logger.error(f"FFmpeg conversion failed (exit code {e.returncode})")
        logger.error(f"Error output:\n{stderr_output}")

        if delete_temp and output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)

        raise HLSConverterError(
            f"Video conversion failed with exit code {e.returncode}. "
            f"Error: {stderr_output[:500]}"
        )

    except Exception as e:
        logger.exception("HLS conversion failed")
        if delete_temp and output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        raise HLSConverterError(str(e))

# -------------------------------------------------------------------
# Batch Processing
# -------------------------------------------------------------------
def batch_convert_to_hls(
    video_list: List[Tuple[str, str]],
    video_bitrate: str = "2000k",
    audio_bitrate: str = "128k",
    hls_time: int = 2,
    crf: int = 23,
    enable_parallel: bool = False,
    parallel_workers: Optional[int] = None
) -> List[Tuple[str, bool, str]]:
    """
    Batch convert multiple videos to HLS using available GPUs

    Args:
        video_list: List of (input_path, output_dir) tuples
        video_bitrate: Video bitrate (default: "2000k")
        audio_bitrate: Audio bitrate (default: "128k")
        hls_time: HLS segment duration in seconds (default: 2)
        crf: Constant Rate Factor for quality (default: 23)
        enable_parallel: Enable parallel segment processing per video (default: False)
        parallel_workers: Number of parallel workers per video (default: auto)

    Returns:
        List of (input_path, success, result) tuples
        where result is the playlist path on success or error message on failure
    """

    gpu_devices = get_available_gpu_devices()
    num_gpus = len(gpu_devices)

    logger.info(f"Batch processing {len(video_list)} videos using {num_gpus} GPU(s)")

    results = []

    # Process videos, distributing across available GPUs
    for idx, (input_path, output_dir) in enumerate(video_list):
        gpu_device = gpu_devices[idx % num_gpus]

        try:
            logger.info(f"Processing video {idx+1}/{len(video_list)}: {input_path} on GPU {gpu_device}")

            playlist = generate_hls(
                input_path=input_path,
                output_dir=output_dir,
                video_bitrate=video_bitrate,
                audio_bitrate=audio_bitrate,
                hls_time=hls_time,
                crf=crf,
                delete_temp=True,
                gpu_device=gpu_device,
                enable_parallel=enable_parallel,
                parallel_workers=parallel_workers
            )

            results.append((input_path, True, playlist))
            logger.info(f"Successfully processed: {input_path}")

        except Exception as e:
            error_msg = f"Failed to process {input_path}: {str(e)}"
            logger.error(error_msg)
            results.append((input_path, False, error_msg))

    # Summary
    successful = sum(1 for _, success, _ in results if success)
    logger.info(f"Batch processing complete: {successful}/{len(video_list)} videos successful")

    return results

# -------------------------------------------------------------------
# Timing Decorator for External Operations
# -------------------------------------------------------------------
def timed_operation(operation_name: str):
    """Decorator to time operations like upload/download"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.info(f"🔄 Starting {operation_name}...")
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.info(f"✅ {operation_name} completed: {elapsed:.2f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"❌ {operation_name} failed after {elapsed:.2f}s: {e}")
                raise
        return wrapper
    return decorator

# -------------------------------------------------------------------
# Public API Wrapper
# -------------------------------------------------------------------
def convert_to_hls(input_video: str, output_dir: str, **kwargs) -> str:
    """
    Convert a single video to HLS format

    Args:
        input_video: Path to input video file
        output_dir: Directory to store HLS output
        **kwargs: Additional parameters (video_bitrate, audio_bitrate, hls_time, crf,
                  gpu_device, enable_parallel, parallel_workers)

    Returns:
        Path to the generated master playlist

    Examples:
        # Basic usage
        playlist = convert_to_hls("input.mp4", "output/")

        # With custom settings
        playlist = convert_to_hls(
            "input.mp4",
            "output/",
            video_bitrate="3000k",
            crf=20,
            gpu_device=0
        )

        # Enable parallel processing for faster encoding
        playlist = convert_to_hls(
            "long_video.mp4",
            "output/",
            enable_parallel=True,
            parallel_workers=4
        )
    """
    return generate_hls(input_video, output_dir, **kwargs)
