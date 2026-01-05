# -- IMPORTS --
import os
import matplotlib
from monai.apps import download_and_extract

# -- CODE --
def setup_pancreas_data(root_dir="data/raw"):
    """Load MSD Pancreas Dataset""" 
    os.makedirs(root_dir, exist_ok = True)
   
    # url to dataset
    resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar"
    md5 = "4f7080cfca169fa8066d17ce6eb061e4"
    
    compressed_file = os.path.join(root_dir, "Task07_Pancreas.tar")
    
    if not os.path.exists(os.path.join(root_dir, "Task07_Pancreas")):
        print(f"[*] Downloading Pancreas Dataset to {root_dir}...")
        download_and_extract(resource, compressed_file, root_dir, md5)
        print("[+] Download and extraction complete.")
    else:
        print("[!] Dataset already exists. Skipping download.")

if __name__ == "__main__":
    setup_pancreas_data()