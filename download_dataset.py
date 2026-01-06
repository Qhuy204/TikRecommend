"""
Script để download TikDataset từ Hugging Face
"""

from huggingface_hub import snapshot_download
from pathlib import Path
import argparse


def download_dataset(output_dir: str = "TikDataset", 
                    repo_id: str = "Qhuy204/TikDataset"):
    """
    Download TikDataset từ Hugging Face Hub
    
    Args:
        output_dir: Thư mục lưu dataset
        repo_id: Repository ID trên Hugging Face
    """
    print("\n" + "="*80)
    print("DOWNLOADING TIKDATASET")
    print("="*80)
    print(f"\nRepository: {repo_id}")
    print(f"Output directory: {output_dir}")
    print("\nThis may take a while depending on your internet speed...")
    print("="*80 + "\n")
    
    try:
        # Download dataset
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=output_dir,
            local_dir_use_symlinks=False,
        )
        
        print("\n" + "="*80)
        print("DOWNLOAD COMPLETE!")
        print("="*80)
        
        # Check downloaded files
        output_path = Path(output_dir)
        files = list(output_path.glob("*"))
        
        print(f"\nDownloaded {len(files)} files to {output_dir}/")
        print("\nSample files:")
        for f in files[:5]:
            print(f"  - {f.name}")
        
        if len(files) > 5:
            print(f"  ... and {len(files) - 5} more files")
        
        print("\n✓ Dataset ready for processing!")
        print("\nNext steps:")
        print("  python recommendation_system.py --mode full --data_dir TikDataset")
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify the repository exists: https://huggingface.co/datasets/Qhuy204/TikDataset")
        print("3. Try updating huggingface_hub: pip install --upgrade huggingface_hub")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download TikDataset')
    parser.add_argument('--output_dir', type=str, default='TikDataset',
                       help='Directory to save dataset')
    parser.add_argument('--repo_id', type=str, default='Qhuy204/TikDataset',
                       help='Hugging Face repository ID')
    
    args = parser.parse_args()
    
    download_dataset(output_dir=args.output_dir, repo_id=args.repo_id)








