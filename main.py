import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import pywt
from PIL import Image
from datetime import datetime
import sys
from collections import deque

# Version info
__version__ = "1.0.0"

class CWTAnalyzer:
    """Continuous Wavelet Transform analyzer for CSV files"""
    
    def __init__(self, scales=128, image_width=1024, image_height=512, log_file=None):
        """
        Initialize CWT analyzer
        
        Args:
            scales: Number of wavelet scales (affects frequency resolution)
            image_width: Output image width in pixels
            image_height: Output image height in pixels
            log_file: Path to log file
        """
        self.scales = np.arange(1, scales + 1)
        self.image_width = image_width
        self.image_height = image_height
        self.wavelet = 'morl'  # Morlet wavelet
        self.log_file = log_file
    
    def log(self, message):
        """Write message to log file"""
        if self.log_file:
            with open(self.log_file, 'a') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{timestamp}] {message}\n")
    
    def perform_cwt(self, data):
        """Perform CWT on a single data channel"""
        coefficients, _ = pywt.cwt(data, self.scales, self.wavelet)
        return np.abs(coefficients)
    
    def normalize_channel(self, data):
        """Normalize data to 0-255 range for image"""
        data_min = data.min()
        data_max = data.max()
        if data_max - data_min == 0:
            return np.zeros_like(data, dtype=np.uint8)
        normalized = (data - data_min) / (data_max - data_min)
        return (normalized * 255).astype(np.uint8)
    
    def create_rgb_image(self, x_data, y_data, z_data):
        """Create RGB image from X, Y, Z CWT coefficients"""
        # Perform CWT on each channel
        x_cwt = self.perform_cwt(x_data)
        y_cwt = self.perform_cwt(y_data)
        z_cwt = self.perform_cwt(z_data)
        
        # Normalize each channel
        r_channel = self.normalize_channel(x_cwt)
        g_channel = self.normalize_channel(y_cwt)
        b_channel = self.normalize_channel(z_cwt)
        
        # Stack into RGB
        rgb_array = np.stack([r_channel, g_channel, b_channel], axis=-1)
        
        # Resize to desired dimensions
        img = Image.fromarray(rgb_array, mode='RGB')
        img = img.resize((self.image_width, self.image_height), Image.LANCZOS)
        
        return img
    
    def process_csv_file(self, csv_path, output_dir):
        """Process a single CSV file"""
        try:
            # Read CSV
            df = pd.read_csv(csv_path)
            
            # Check for required columns
            if not all(col in df.columns for col in ['X', 'Y', 'Z']):
                self.log(f"SKIPPED: {csv_path} - Missing X, Y, or Z columns")
                return False
            
            # Create RGB CWT image
            img = self.create_rgb_image(
                df['X'].values,
                df['Y'].values,
                df['Z'].values
            )
            
            # Save image
            output_path = output_dir / f"{csv_path.stem}_cwt.png"
            img.save(output_path)
            self.log(f"SUCCESS: {csv_path} -> {output_path}")
            return True
            
        except Exception as e:
            self.log(f"ERROR: {csv_path} - {str(e)}")
            return False


def clear_lines(n):
    """Clear n lines from terminal"""
    for _ in range(n):
        sys.stdout.write('\033[F')  # Move cursor up
        sys.stdout.write('\033[K')  # Clear line


def display_progress(current, total, current_file, recent_files, lines_to_clear=0):
    """Display progress bar, current file, and recent files"""
    # Clear previous output
    if lines_to_clear > 0:
        clear_lines(lines_to_clear)
    
    # Calculate progress
    percent = (current / total) * 100
    bar_length = 40
    filled = int(bar_length * current / total)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    # Build output
    lines = []
    
    # Show recent completed files (last 5)
    lines.append("")
    for file in recent_files:
        lines.append(f"  ✓ {file.name}")
    
    # Show current file
    lines.append("")
    lines.append(f"  ▶ {current_file.name}")
    
    # Show progress bar
    lines.append("")
    lines.append(f"Progress: [{bar}] {current}/{total} ({percent:.1f}%)")
    
    # Print all lines
    for line in lines:
        print(line)
    
    sys.stdout.flush()
    
    # Return number of lines printed for next clear
    return len(lines)


def process_path(input_path, analyzer):
    """Process a file or directory"""
    input_path = Path(input_path)
    output_base = Path('outputs')
    
    if input_path.is_file() and input_path.suffix == '.csv':
        # Single file
        output_dir = output_base
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing 1 CSV file")
        
        recent_files = deque(maxlen=5)
        lines_printed = display_progress(1, 1, input_path, recent_files)
        success = analyzer.process_csv_file(input_path, output_dir)
        
    elif input_path.is_dir():
        # Directory - search recursively
        csv_files = list(input_path.rglob('*.csv'))
        
        if not csv_files:
            print("No CSV files found")
            analyzer.log("No CSV files found in directory")
            return
        
        total = len(csv_files)
        analyzer.log(f"Found {total} CSV file(s) to process")
        
        # Print this once and don't clear it
        print(f"Found {total} CSV files")
        
        recent_files = deque(maxlen=5)
        lines_printed = 0
        
        for idx, csv_file in enumerate(csv_files, 1):
            # Display progress with current file
            lines_printed = display_progress(idx, total, csv_file, recent_files, lines_printed)
            
            # Maintain folder structure
            relative_path = csv_file.relative_to(input_path)
            output_dir = output_base / relative_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            success = analyzer.process_csv_file(csv_file, output_dir)
            if success:
                recent_files.append(csv_file)
    
    else:
        print("Error: Path must be a CSV file or directory")
        analyzer.log("ERROR: Invalid path - must be CSV file or directory")


def main():
    # Custom description with examples
    description = """
CWT Analyzer - Generate RGB scalogram images from CSV files using Continuous Wavelet Transform.

The program maps X, Y, Z columns to R, G, B channels respectively.
All outputs are saved to the 'outputs/' folder maintaining the original folder structure.
    """
    
    epilog = """
Examples:
  # Process a single CSV file
  python main.py data.csv
  
  # Process all CSV files in a directory
  python main.py /path/to/folder
  
  # Custom image resolution
  python main.py data.csv --width 2048 --height 1024
  
  # Custom frequency resolution
  python main.py data.csv --scales 256
  
  # All custom parameters
  python main.py /path/to/folder --scales 256 --width 2048 --height 1024

CSV Requirements:
  - Must contain columns named: X, Y, Z
  - First column is treated as time/index
  
Output:
  - RGB PNG images: outputs/[path]/*_cwt.png
  - Processing log: outputs/log.txt
  
For more information, visit: https://github.com/yourusername/cwt-gen
    """
    
    parser = argparse.ArgumentParser(
        prog='cwt-gen',
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add version flag
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )
    
    # Required argument
    parser.add_argument(
        'path',
        type=str,
        help='Path to CSV file or directory containing CSV files'
    )
    
    # Optional arguments
    parser.add_argument(
        '--scales',
        type=int,
        default=128,
        metavar='N',
        help='Number of wavelet scales for frequency resolution (default: 128)'
    )
    
    parser.add_argument(
        '--width',
        type=int,
        default=1024,
        metavar='PIXELS',
        help='Output image width in pixels (default: 1024)'
    )
    
    parser.add_argument(
        '--height',
        type=int,
        default=512,
        metavar='PIXELS',
        help='Output image height in pixels (default: 512)'
    )
    
    args = parser.parse_args()
    
    # Setup log file
    output_base = Path('outputs')
    output_base.mkdir(parents=True, exist_ok=True)
    log_file = output_base / 'log.txt'
    
    # Write header to log
    with open(log_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"CWT Analyzer v{__version__}\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Input path: {args.path}\n")
        f.write(f"Scales: {args.scales}\n")
        f.write(f"Image dimensions: {args.width}x{args.height}\n")
        f.write(f"Wavelet: Morlet (morl)\n")
        f.write("=" * 80 + "\n\n")
    
    # Initialize analyzer with specified parameters
    analyzer = CWTAnalyzer(
        scales=args.scales,
        image_width=args.width,
        image_height=args.height,
        log_file=log_file
    )
    
    # Process the path
    process_path(args.path, analyzer)
    
    # Write completion to log
    analyzer.log("=" * 80)
    analyzer.log(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    analyzer.log("=" * 80)
    
    print("\n\nProcessing complete!")
    print(f"Log saved to: {log_file}")


if __name__ == '__main__':
    main()