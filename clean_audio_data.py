import os
import sys
from pathlib import Path
import argparse
import librosa
import logging
import selectors
import fcntl

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Error message fragment to detect
LIBMPG123_ERROR_MSG = "part2_3_length"

def check_audio_file(file_path: Path) -> bool:
    """
    Tries to load an audio file and checks for libmpg123 errors on stderr 
    using low-level file descriptor manipulation.

    Args:
        file_path: Path object for the audio file.

    Returns:
        True if the file loads without the specific error, False otherwise.
    """
    # Create a pipe to capture stderr
    read_fd, write_fd = os.pipe()
    # Make the read end non-blocking
    fcntl.fcntl(read_fd, fcntl.F_SETFL, os.O_NONBLOCK)

    # Save the original stderr descriptor
    original_stderr_fd = os.dup(sys.stderr.fileno())
    # Duplicate the write end of the pipe onto stderr
    os.dup2(write_fd, sys.stderr.fileno())

    stderr_output = ""
    detected_error = False

    try:
        # Attempt to load the file - stderr now goes to the pipe
        librosa.load(file_path, sr=None, duration=0.1)
    except Exception as e:
        # Catch Python-level loading errors
        logger.error(f"Python exception loading file {file_path}: {e}")
        detected_error = True # Treat other load errors as bad files too
    finally:
        # Restore original stderr *before* reading from the pipe
        os.dup2(original_stderr_fd, sys.stderr.fileno())
        # Close the duplicated descriptors
        os.close(write_fd)
        os.close(original_stderr_fd)
        
        # Read the captured stderr from the pipe
        captured_stderr_bytes = b''
        while True:
            try:
                chunk = os.read(read_fd, 4096) # Read in chunks
                if not chunk:
                    break
                captured_stderr_bytes += chunk
            except BlockingIOError:
                # No more data to read immediately
                break
            except Exception as read_err:
                 logger.error(f"Error reading from stderr pipe for {file_path}: {read_err}")
                 break # Avoid infinite loop on other errors
        os.close(read_fd)
        
        try:
            stderr_output = captured_stderr_bytes.decode('utf-8', errors='ignore')
        except Exception as decode_err:
            logger.error(f"Error decoding stderr output for {file_path}: {decode_err}")

    # --- Check the captured output --- 
    if not detected_error: # Only check stderr string if no Python exception occurred
        if LIBMPG123_ERROR_MSG in stderr_output:
            logger.warning(f"Detected libmpg123 error string in file: {file_path}")
            # Uncomment to log the actual captured error message:
            # logger.debug(f"Captured stderr for {file_path}:\n>>>\n{stderr_output.strip()}\n<<<" )
            detected_error = True
        elif stderr_output: # Log if *any* stderr was captured but didn't match
             logger.debug(f"Captured non-matching stderr for {file_path}:\n>>>\n{stderr_output.strip()}\n<<<" )

    # Return True if no error was detected, False otherwise
    return not detected_error

def main(data_dir_str: str):
    """
    Scans the data directory, identifies bad audio files, asks for confirmation, 
    and deletes them if approved.
    """
    data_dir = Path(data_dir_str)
    project_root = Path(os.getcwd()) # Assume script is run from project root
    
    # Ensure data_dir is relative to the project root if not absolute
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir_str
        
    if not data_dir.is_dir():
        logger.error(f"Error: Data directory not found or is not a directory: {data_dir}")
        sys.exit(1)

    logger.info(f"Scanning directory: {data_dir} for problematic .wav files...")
    
    potential_files = list(data_dir.rglob('*.wav')) # Use rglob for recursive search
    if not potential_files:
        logger.info("No .wav files found in the directory.")
        sys.exit(0)
        
    logger.info(f"Found {len(potential_files)} potential .wav files. Checking each file...")

    bad_files_to_delete = []
    checked_count = 0
    error_count = 0

    for file_path in potential_files:
        checked_count += 1
        if not check_audio_file(file_path):
            bad_files_to_delete.append(file_path)
            error_count += 1
        if checked_count % 100 == 0:
             logger.info(f"Checked {checked_count}/{len(potential_files)} files... ({error_count} problematic files found so far)")

    logger.info(f"Scan complete. Checked {checked_count} files.")

    if not bad_files_to_delete:
        logger.info("No problematic audio files detected.")
        sys.exit(0)

    print("\nThe following potentially problematic files were detected:")
    for file_path in bad_files_to_delete:
        # Display path relative to the original data_dir for clarity
        try:
            display_path = file_path.relative_to(data_dir)
        except ValueError:
             display_path = file_path # Fallback to absolute if error
        print(f"- {display_path}  (Located at: {file_path})")

    print(f"\nFound {len(bad_files_to_delete)} potentially problematic files.")
    
    # Ask for confirmation
    try:
        confirm = input("Do you want to permanently delete these files? (yes/No): ").strip().lower()
    except EOFError:
        confirm = 'no' # Treat EOF (e.g., script piped input) as no

    if confirm in ['y', 'yes']:
        print("\nProceeding with deletion...")
        deleted_count = 0
        deletion_error_count = 0
        for file_path in bad_files_to_delete:
            try:
                file_path.unlink()
                logger.info(f"Deleted: {file_path}")
                deleted_count += 1
            except Exception as e:
                logger.error(f"Error deleting {file_path}: {e}")
                deletion_error_count += 1
        print(f"\nDeletion complete. {deleted_count} files deleted, {deletion_error_count} errors during deletion.")
    else:
        print("\nDeletion cancelled. No files were deleted.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan an audio directory for problematic .wav files (specifically libmpg123 errors) and optionally delete them.")
    parser.add_argument("data_directory", type=str, help="Path to the directory containing audio files to scan (e.g., 50_speakers_audio_data)")
    
    args = parser.parse_args()
    main(args.data_directory) 