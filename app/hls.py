import os
import subprocess
import shutil
import multiprocessing
from pathlib import Path
from typing import Dict, Optional

class HLSConverterError(Exception):
    pass

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
            check=True
        )

        import json
        data = json.loads(result.stdout)

        if not data.get("streams"):
            raise HLSConverterError("No video stream found")

        stream = data["streams"][0]
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        duration = float(stream.get("duration", 0))
        num, den = map(float, stream.get("r_frame_rate", "0/1").split("/"))
        fps = num / den if den else 30

        return {
            "width": width,
            "height": height,
            "duration": duration,
            "fps": fps,
            "codec": stream.get("codec_name", "h264")
        }

    except Exception as e:
        raise HLSConverterError(f"Video info error: {str(e)}")

def get_hardware_accel() -> Optional[str]:
    """Detect available hardware acceleration."""
    try:
        # Check for NVIDIA
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if result.returncode == 0:
            return "h264_nvenc"
            
        # Check for Intel Quick Sync
        result = subprocess.run(
            ["vainfo"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if result.returncode == 0 and b'VAEntrypointEncSlice' in result.stdout:
            return "h264_vaapi"
            
        # Check for AMD AMF
        try:
            import platform
            if platform.system() == 'Windows':
                import winreg
                try:
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\AMD\AMF"):
                        return "h264_amf"
                except WindowsError:
                    pass
        except ImportError:
            pass
            
        return None
    except Exception:
        return None

def generate_hls(
    input_path: str,
    output_dir: str,
    video_bitrate: str = "2000k",
    audio_bitrate: str = "128k",
    hls_time: int = 2,
    preset: str = "veryfast",
    delete_temp: bool = True,
    use_hw_accel: bool = True,
    crf: int = 23
) -> str:
    """
    Generate HLS stream with optimized settings for speed.
    
    Args:
        input_path: Path to input video file
        output_dir: Directory to save HLS output
        video_bitrate: Target video bitrate (default: 2000k)
        audio_bitrate: Target audio bitrate (default: 128k)
        hls_time: Target segment length in seconds (default: 2)
        preset: Encoding preset (default: veryfast)
        delete_temp: Whether to delete temp files on error (default: True)
        use_hw_accel: Enable hardware acceleration if available (default: True)
        crf: Constant Rate Factor (lower = better quality, 18-28 is good range)
    """
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise HLSConverterError("FFmpeg not installed")

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = Path(output_dir)
    segment_dir = output_dir / "segment"
    
    # Get CPU count for thread optimization
    cpu_count = min(multiprocessing.cpu_count(), 8)  # Cap at 8 to avoid diminishing returns
    
    # Detect hardware acceleration
    hw_accel = None
    hw_accel_params = []
    if use_hw_accel:
        hw_accel = get_hardware_accel()
        if hw_accel:
            hw_accel_params = ["-c:v", hw_accel, "-b:v", video_bitrate]
            preset = "fast"  # Hardware encoders often don't benefit from slower presets

    try:
        video_info = get_video_info(input_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        segment_dir.mkdir(parents=True, exist_ok=True)

        output_playlist = segment_dir / "playlist.m3u8"
        master_playlist = output_dir / "playlist.m3u8"
        
        fps = round(video_info["fps"]) if video_info["fps"] else 30
        gop = fps * hls_time
        
        # Build the FFmpeg command
        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-threads", str(cpu_count),
            "-cpu-used", "1" if hw_accel else "0",
            "-r", str(fps),
            "-vsync", "cfr",
            "-force_key_frames", f"expr:gte(t,n_forced*{hls_time})",
        ]
        
        # Video encoding parameters
        cmd.extend([
            "-c:v", hw_accel if hw_accel else "libx264",
            "-preset", "superfast" if not hw_accel else "fast",
            "-tune", "fastdecode,zerolatency",
            "-movflags", "+faststart",
            "-g", str(gop),
            "-keyint_min", str(gop),
            "-sc_threshold", "0",
            "-pix_fmt", "yuv420p",
            "-x264opts", "no-mbtree:sliced-threads=0:weightp=0:no-deblock=1:rc-lookahead=0:ref=1:subme=0:me=dia:merange=16:trellis=0",
            "-threads", str(cpu_count)
        ])
        
        # Add bitrate control based on whether using HW acceleration
        if hw_accel:
            cmd.extend([
                "-b:v", video_bitrate,
                "-maxrate", video_bitrate,
                "-bufsize", str(int(video_bitrate[:-1]) * 2) + video_bitrate[-1],
                "-qmin", "23",
                "-qmax", "35"
            ])
        else:
            cmd.extend([
                "-crf", str(crf),
                "-maxrate", video_bitrate,
                "-bufsize", str(int(video_bitrate[:-1]) * 2) + video_bitrate[-1],
                "-x264opts", "no-scenecut:no-mbtree:sliced-threads=1:weightp=0:no-deblock=1:rc-lookahead=0:ref=1:subme=0:me=dia:merange=16:trellis=0"
            ])
        
        # Audio parameters
        cmd.extend([
            "-c:a", "aac",
            "-b:a", audio_bitrate,
            "-ar", "48000",
            "-ac", "2",
            "-aac_coder", "twoloop",
            "-profile:a", "aac_low"
        ])
        
        # HLS parameters
        segment_pattern = "segment_%03d.ts"
        cmd.extend([
            "-f", "hls",
            "-hls_time", str(hls_time),
            "-hls_playlist_type", "vod",
            "-hls_segment_type", "mpegts",
            "-hls_flags", "independent_segments+delete_segments",
            "-hls_list_size", "0",
            "-start_number", "0",
            "-hls_segment_filename", str(segment_dir / segment_pattern),
            "-hls_base_url", "",  # Remove any base URL to prevent path duplication
            str(output_playlist)
        ])
        
        # Create a custom master playlist with correct paths
        master_playlist_content = (
            "#EXTM3U\n"
            "#EXT-X-VERSION:3\n"
            f"#EXT-X-STREAM-INF:BANDWIDTH=2000000,RESOLUTION={video_info['width']}x{video_info['height']}\n"
            f"segment/playlist.m3u8\n"
        )
        
        # Run the command with Popen for better control
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line.strip())
            
        process.wait()
        
        if process.returncode != 0:
            raise HLSConverterError(f"FFmpeg failed with return code {process.returncode}")
            
        if not output_playlist.exists():
            raise HLSConverterError("HLS generation failed - no playlist created")
            
        # Create master playlist with correct paths
        with open(master_playlist, "w") as f:
            f.write(master_playlist_content)
            
        return str(master_playlist)

    except Exception as e:
        if delete_temp and output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        raise HLSConverterError(f"HLS conversion failed: {str(e)}") from e

def convert_to_hls(input_video: str, output_dir: str, **kwargs) -> str:
    """Convert a video file to HLS format with optimized settings."""
    return generate_hls(input_video, output_dir, **kwargs)

if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='Convert video to HLS format with optimized settings')
    parser.add_argument('input', help='Input video file')
    parser.add_argument('output', help='Output directory for HLS files')
    parser.add_argument('--bitrate', default='2000k', help='Video bitrate (default: 2000k)')
    parser.add_argument('--audio-bitrate', default='128k', help='Audio bitrate (default: 128k)')
    parser.add_argument('--segment-time', type=int, default=2, help='Segment duration in seconds (default: 2)')
    parser.add_argument('--preset', default='veryfast', 
                       choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium'],
                       help='Encoding preset (default: veryfast)')
    parser.add_argument('--crf', type=int, default=23, 
                       help='Constant Rate Factor (lower = better quality, 18-28 is good range, default: 23)')
    parser.add_argument('--no-hw-accel', action='store_false', dest='use_hw_accel',
                       help='Disable hardware acceleration')
    
    args = parser.parse_args()
    
    try:
        print(f"Starting HLS conversion with {'hardware' if args.use_hw_accel else 'software'} acceleration...")
        result = convert_to_hls(
            args.input,
            args.output,
            video_bitrate=args.bitrate,
            audio_bitrate=args.audio_bitrate,
            hls_time=args.segment_time,
            preset=args.preset,
            crf=args.crf,
            use_hw_accel=args.use_hw_accel
        )
        print(f"\nConversion complete! Master playlist: {result}")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)