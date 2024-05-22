import os
import shutil
import zipfile
import gdown
import argparse

def download_from_gdrive(obj_dir, json_dir):
    print("Dental Meshes #1")
    gdown.download("https://drive.google.com/uc?id=1N_ZewZ1yUTDKo3fe7ZLGgUho8_iWrFbi")
    with zipfile.ZipFile("3D_scans_per_patient_obj_files.zip", "r") as zip_ref:
        zip_ref.extractall(obj_dir)
    os.remove("3D_scans_per_patient_obj_files.zip")

    print("Dental Meshes #2")
    gdown.download("https://drive.google.com/uc?id=140qWvJR3zV-6j-A2uzi1vBfPa9M87HaA")
    with zipfile.ZipFile("3D_scans_per_patient_obj_files_b2.zip", "r") as zip_ref:
        zip_ref.extractall(obj_dir)
    os.remove("3D_scans_per_patient_obj_files_b2.zip")

    print("Ground Truth Labels #1")
    gdown.download("https://drive.google.com/uc?id=1x9Euejg6RL_9HXgdNYmqRtV2Em6YuFtr")
    with zipfile.ZipFile("ground-truth_labels_instances.zip", "r") as zip_ref:
        zip_ref.extractall(json_dir)
    os.remove("ground-truth_labels_instances.zip")

    print("Ground Truth Labels #2")
    gdown.download("https://drive.google.com/uc?id=1GKjz4YaoXlXwsvPCVKXT07-AstLQ6u-y")
    with zipfile.ZipFile("ground-truth_labels_instances_b2.zip", "r") as zip_ref:
        zip_ref.extractall(json_dir)
    os.remove("ground-truth_labels_instances_b2.zip")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_dir", type=str, default="data_obj_parent_directory", help="Directory to save the dental mesh files")
    parser.add_argument("--json_dir", type=str, default="data_json_parent_directory", help="Directory to save the ground truth files")
    parser.add_argument("--force_download", action=argparse.BooleanOptionalAction, help="Re-download the data")
    args = parser.parse_args()
    
    if os.path.exists(args.obj_dir):
        if args.force_download:
            shutil.rmtree(args.obj_dir, ignore_errors=True)
            shutil.rmtree(args.json_dir, ignore_errors=True)
            download_from_gdrive(args.obj_dir, args.json_dir)
        else:
            print("Data already exists. Use --force_download to re-download.")
    else:
        download_from_gdrive(args.obj_dir, args.json_dir)
    
if __name__ == "__main__":
    main()