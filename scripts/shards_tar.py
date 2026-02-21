import tarfile
import os
import math

# --- Configuration ---
SOURCE_TAR = r"C:\Users\rajit\Downloads\shoptalk-ai\data\abo-images-small.tar"
OUTPUT_DIR = r"C:\Users\rajit\Downloads\shoptalk-ai\data\shards"
NUM_SHARDS = 30

def split_tar():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Opening source: {SOURCE_TAR}")
    
    # 1. Get list of all members to calculate distribution
    with tarfile.open(SOURCE_TAR, "r") as src:
        members = src.getmembers()
        total_files = len(members)
        files_per_shard = math.ceil(total_files / NUM_SHARDS)
        
        print(f"Total files found: {total_files}")
        print(f"Targeting ~{files_per_shard} files per shard.")

        # 2. Iterate through shards
        for shard_idx in range(NUM_SHARDS):
            shard_name = os.path.join(OUTPUT_DIR, f"shard_{shard_idx:04d}.tar")
            start_idx = shard_idx * files_per_shard
            end_idx = min(start_idx + files_per_shard, total_files)
            
            if start_idx >= total_files:
                break

            print(f"Writing {shard_name} (files {start_idx} to {end_idx})...")
            
            with tarfile.open(shard_name, "w") as dest:
                for i in range(start_idx, end_idx):
                    member = members[i]
                    # Extract file object from source and add to destination
                    f = src.extractfile(member)
                    if f is not None:
                        dest.addfile(member, f)

    print("\nSuccessfully split tar into 30 shards!")

if __name__ == "__main__":
    split_tar()