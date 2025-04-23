import os
import zipfile
import requests
from tqdm import tqdm
import subprocess

class DataDownloader:
    def __init__(self, url, output_path):
        self.url = url
        self.output_path = output_path
        
    def download(self):
        """Download the dataset from Kaggle"""
        print(f"Downloading dataset from {self.url}")
        
        # Try using Kaggle API first
        try:
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", "mohitsingh1804/plantvillage", 
                 "-p", os.path.dirname(self.output_path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print("Download complete using Kaggle API")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print("Kaggle API not available, falling back to direct download")
            
            # Fallback to direct download
            try:
                response = requests.get(self.url, stream=True, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                })
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                block_size = 8192
                progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
                
                with open(self.output_path, 'wb') as f:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        f.write(data)
                progress_bar.close()
                
                if total_size != 0 and progress_bar.n != total_size:
                    print("Download failed: incomplete download")
                    return False
                
                print("Download complete using direct download")
                return True
            except Exception as e:
                print(f"Download failed: {str(e)}")
                return False

class DataOrganizer:
    def __init__(self, zip_path, output_dir):
        self.zip_path = zip_path
        self.output_dir = output_dir
        
    def extract_zip(self):
        """Extract the zip file to the output directory"""
        if not os.path.exists(self.zip_path):
            raise FileNotFoundError(f"Zip file not found at {self.zip_path}")
            
        print(f"Extracting {self.zip_path} to {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                # Get list of files with progress
                file_list = zip_ref.infolist()
                with tqdm(total=len(file_list), desc="Extracting") as pbar:
                    for file in file_list:
                        zip_ref.extract(file, self.output_dir)
                        pbar.update(1)
            
            print("Extraction complete")
            
            # The dataset is already organized into train/val
            # Verify the structure
            train_path = os.path.join(self.output_dir, "train")
            val_path = os.path.join(self.output_dir, "val")
            
            if not os.path.exists(train_path) or not os.path.exists(val_path):
                print("Warning: Extracted dataset doesn't contain standard train/val directories")
            
            return True
        except Exception as e:
            print(f"Extraction failed: {str(e)}")
            return False
    
    def cleanup(self, paths_to_remove):
        """Clean up temporary files"""
        print("Cleaning up...")
        for path in paths_to_remove:
            try:
                if os.path.isfile(path):
                    os.remove(path)
                    print(f"Removed file: {path}")
                elif os.path.isdir(path):
                    print(f"Keeping directory: {path} (contains dataset)")
            except Exception as e:
                print(f"Failed to remove {path}: {str(e)}")
        print("Cleanup complete")

def main():
    # Configuration
    dataset_url = "https://www.kaggle.com/api/v1/datasets/download/mohitsingh1804/plantvillage"
    download_path = "./plantvillage.zip"
    output_dir = "./training/"  # This will contain train/val subdirectories
    
    try:
        # Step 1: Download the dataset
        downloader = DataDownloader(dataset_url, download_path)
        if not downloader.download():
            raise RuntimeError("Dataset download failed")
        
        # Step 2: Extract directly to the output directory
        organizer = DataOrganizer(download_path, output_dir)
        if not organizer.extract_zip():
            raise RuntimeError("Dataset extraction failed")
        
        # Step 3: Cleanup (only remove zip file, keep extracted directories)
        organizer.cleanup([download_path])
        
        print("\nDataset preparation completed successfully!")
        print(f"Training data available at: {os.path.join(output_dir, 'train')}")
        print(f"Validation data available at: {os.path.join(output_dir, 'val')}")
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Dataset preparation failed")
        exit(1)

if __name__ == "__main__":
    main()