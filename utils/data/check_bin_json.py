import os

def find_bin_without_json(directory):
    bin_files = [f for f in os.listdir(directory) if f.endswith('.bin')]
    json_files = [f.replace('.json', '') for f in os.listdir(directory) if f.endswith('.json')]

    bin_without_json = [f for f in bin_files if f.replace('.bin', '') not in json_files]

    return bin_without_json

def find_json(directory):
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    json_files.sort()
    return json_files

if __name__ == "__main__":
    directory = 'data/cpg'
    bin_without_json = find_bin_without_json(directory)
    
    with open('bin_without_json.txt', 'w') as f:
        for item in bin_without_json:
            f.write("%s\n" % item)