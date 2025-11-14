import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import pywt
from PIL import Image
from datetime import datetime
import sys
from collections import deque

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


def save_cursor_position():
    """Save current cursor position"""
    sys.stdout.write('\033[s')
    sys.stdout.flush()


def restore_cursor_position():
    """Restore cursor to saved position"""
    sys.stdout.write('\033[u')
    sys.stdout.flush()


def clear_from_cursor():
    """Clear from cursor to end of screen"""
    sys.stdout.write('\033[J')
    sys.stdout.flush()


def display_progress(current, total, current_file, recent_files, first_call=False):
    """Display progress bar, current file, and recent files"""
    if not first_call:
        # Restore cursor to saved position and clear everything below
        restore_cursor_position()
        clear_from_cursor()
    else:
        # Save cursor position on first call (marks the starting point)
        save_cursor_position()
    
    # Calculate progress
    percent = (current / total) * 100
    bar_length = 40
    filled = int(bar_length * current / total)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    # Show recent completed files (last 5)
    print()
    for file in recent_files:
        print(f"  ✓ {file.name}")
    
    # Show current file
    print()
    print(f"  ▶ {current_file.name}")
    
    # Show progress bar
    print()
    print(f"Progress: [{bar}] {current}/{total} ({percent:.1f}%)")
    
    sys.stdout.flush()


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
        display_progress(1, 1, input_path, recent_files, first_call=True)
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
        first_call = True
        
        for idx, csv_file in enumerate(csv_files, 1):
            # Display progress with current file
            display_progress(idx, total, csv_file, recent_files, first_call=first_call)
            first_call = False
            
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
    parser = argparse.ArgumentParser(
        description='Perform CWT on CSV files and generate RGB images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('path', type=str, help='Path to CSV file or directory')
    parser.add_argument('--scales', type=int, default=128, 
                       help='Number of wavelet scales (frequency resolution)')
    parser.add_argument('--width', type=int, default=1024,
                       help='Output image width in pixels')
    parser.add_argument('--height', type=int, default=512,
                       help='Output image height in pixels')
    
    args = parser.parse_args()
    
    # Setup log file
    output_base = Path('outputs')
    output_base.mkdir(parents=True, exist_ok=True)
    log_file = output_base / 'log.txt'
    
    # Write header to log
    with open(log_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CWT Analyzer Log\n")
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