import os
import boto3
import logging
import concurrent.futures
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from functools import partial

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WasabiUploader:
    def __init__(self):
        load_dotenv()
        self.access_key   = os.getenv('WASABI_ACCESS_KEY')
        self.secret_key   = os.getenv('WASABI_SECRET_KEY')
        self.endpoint_url = os.getenv('WASABI_ENDPOINT_URL')
        self.region       = os.getenv('WASABI_REGION')
     
        missing_vars = []
        if not self.access_key:
            missing_vars.append('WASABI_ACCESS_KEY')
        if not self.secret_key:
            missing_vars.append('WASABI_SECRET_KEY')
        if not self.endpoint_url:
            missing_vars.append('WASABI_ENDPOINT_URL')
        if not self.region:
            missing_vars.append('WASABI_REGION')
        if missing_vars:
            raise ValueError(f"Missing Required Environment Variables: {', '.join(missing_vars)}")
        try:
            self.s3_client = boto3.client(
                's3',
                endpoint_url          = self.endpoint_url,
                aws_access_key_id     = self.access_key,
                aws_secret_access_key = self.secret_key,
                region_name           = self.region
            )
        except Exception as e:
            raise ConnectionError(f"Failed To Initialize Wasabi Client: {str(e)}")

    def _parse_destination_path(self, destination_path: str) -> Tuple[str, str]:
        if not destination_path:
            raise ValueError("Destination Path Is Required")

        parts = destination_path.strip('/').split('/', 1)
        if len(parts) < 1:
            raise ValueError("Invalid Destination Path Format. Expected: 'bucket-name/folder/path'")
            
        bucket_name = parts[0]
        folder_path = parts[1] if len(parts) > 1 else ''
        
        if not bucket_name:
            raise ValueError("Bucket Name Cannot Be Empty")
            
        return bucket_name, folder_path

    def _upload_single_file(self, file_path: Path, bucket_name: str, s3_key: str) -> tuple[str, str]:
        """Upload a single file to Wasabi S3"""
        try:
            self.s3_client.upload_file(
                str(file_path),
                bucket_name,
                s3_key,
                ExtraArgs={
                    'ACL': 'public-read',
                    'ContentType': self._get_content_type(file_path)
                }
            )
            file_url = f"https://{bucket_name}.s3.{self.region}.wasabisys.com/{s3_key}"
            return str(file_path.name), file_url
        except Exception as e:
            logger.error(f"Failed to upload {file_path}: {str(e)}")
            raise

    def upload_folder(self, local_folder: str, destination_path: str, max_workers: int = 5) -> Dict[str, str]:
        """
        Upload a folder to Wasabi S3 with parallel uploads
        
        Args:
            local_folder: Path to the local folder to upload
            destination_path: Destination path in format 'bucket-name/folder/path'
            max_workers: Maximum number of parallel uploads (default: 5)
            
        Returns:
            Dict mapping relative file paths to their S3 URLs
        """
        if not os.path.isdir(local_folder):
            raise FileNotFoundError(f"Local folder not found: {local_folder}")
       
        bucket_name, folder_path = self._parse_destination_path(destination_path)
        folder_path = folder_path.strip('/')
        uploaded_files = {}
        local_folder = Path(local_folder).resolve()
        
        # Collect all files to upload
        file_paths = []
        for root, _, files in os.walk(local_folder):
            for file in files:
                file_paths.append(Path(root) / file)
        
        if not file_paths:
            return {}
        
        # Create a thread pool for parallel uploads
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Prepare upload tasks
            upload_tasks = []
            for file_path in file_paths:
                relative_path = file_path.relative_to(local_folder)
                s3_key = str(Path(folder_path) / relative_path) if folder_path else str(relative_path)
                upload_tasks.append(
                    executor.submit(
                        self._upload_single_file,
                        file_path=file_path,
                        bucket_name=bucket_name,
                        s3_key=s3_key
                    )
                )
            
            # Process completed uploads
            for future in concurrent.futures.as_completed(upload_tasks):
                try:
                    filename, file_url = future.result()
                    uploaded_files[filename] = file_url
                    logger.info(f"Uploaded: {filename}")
                except Exception as e:
                    logger.error(f"Upload failed: {str(e)}")
        
        return uploaded_files

    def upload_directory(self, local_path: str, destination_path: str) -> Dict[str, str]:
        """Alias for upload_folder to maintain backward compatibility."""
        return self.upload_folder(local_path, destination_path)

    def _get_content_type(self, file_path: Path) -> str:
        """Get MIME type for a file based on its extension"""
        ext = file_path.suffix.lower()
        content_types = {
            # Video
            '.m3u8': 'application/x-mpegURL',
            '.m3u': 'application/x-mpegURL',
            '.ts': 'video/MP2T',
            '.m2ts': 'video/MP2T',
            '.mp4': 'video/mp4',
            '.webm': 'video/webm',
            '.mov': 'video/quicktime',
            '.avi': 'video/x-msvideo',
            '.mkv': 'video/x-matroska',
            
            # Audio
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.ogg': 'audio/ogg',
            '.m4a': 'audio/mp4',
            
            # Images
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.svg': 'image/svg+xml',
            
            # Documents
            '.pdf': 'application/pdf',
            '.doc': 'application/msword',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.xls': 'application/vnd.ms-excel',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.ppt': 'application/vnd.ms-powerpoint',
            '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            
            # Text
            '.txt': 'text/plain',
            '.csv': 'text/csv',
            '.html': 'text/html',
            '.css': 'text/css',
            '.js': 'application/javascript',
            '.json': 'application/json',
            '.xml': 'application/xml',
            
            # Archives
            '.zip': 'application/zip',
            '.rar': 'application/x-rar-compressed',
            '.7z': 'application/x-7z-compressed',
            '.tar': 'application/x-tar',
            '.gz': 'application/gzip',
        }
        return content_types.get(ext, 'binary/octet-stream')

    def get_file_url(self, destination_path: str, file_key: str = '') -> str:
        bucket_name, folder_path = self._parse_destination_path(destination_path)
        path_parts               = [part for part in [folder_path, file_key] if part]
        full_path                = '/'.join(path_parts)
        return f"https://{bucket_name}.s3.{self.region}.wasabisys.com/{full_path}"