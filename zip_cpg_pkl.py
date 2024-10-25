import os
import zipfile

def zip_pkl_files(source_dir, output_filename):
    with zipfile.ZipFile(output_filename, 'w') as zipf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.endswith('.pkl'):
                    zipf.write(os.path.join(root, file), arcname=file)

if __name__ == "__main__":
    source_directory = "data/cpg"
    output_zipfile = "cpg.zip"
    zip_pkl_files(source_directory, output_zipfile)
    print(f"Created {output_zipfile} containing all .pkl files from {source_directory}")