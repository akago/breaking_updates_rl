import re
import logging
import pandas as pd 

def extract_hash_all_passed_patch_from_log(file_path: str)-> list:
    """
    Extract the hash of the all-passed patch from the log file.
    param: file_path: path to the log file
    returns: list of hashes of the all-passed patch or empty list if not found
    """
    passed_hash = []
    # pattern with group for hash
    PATTERN = re.compile(r"Patches for breaking commit ([a-zA-Z0-9]+) passed all tests.")
    try:
        with open(file_path, 'r') as f:
            for line in f:
                match = PATTERN.search(line)
                if match:
                    patch_hash = match.group(1)
                    passed_hash.append(patch_hash)
    except Exception as e:
        logging.error(f"Error reading log file {file_path}: {e}")
    return passed_hash
    
    
def merge_infos(df_per_file_status:pd.DataFrame, success_breaking_commits_list:list) -> pd.DataFrame:
    for idx, row in df_per_file_status.iterrows():
        commit_hash = row['breakingCommit']
        if commit_hash in success_breaking_commits_list:
            df_per_file_status.at[idx, 'build_success'] = True
        else:
            df_per_file_status.at[idx, 'build_success'] = False
    # change the name of column 'accepted' to 'file_success'
    df_per_file_status = df_per_file_status.rename(columns={'accepted': 'file_success'})
    return df_per_file_status

def main():
    path_to_filter_log = "/home/xchen6/breaking_updates_rl/filter_distillation.sh_18560319.out"
    commit_hash_list = extract_hash_all_passed_patch_from_log(path_to_filter_log)
    
    logging.info(f"Number of passing fixes of breaking commit: {len(commit_hash_list)}")
    
    path_to_csv_info = "/home/xchen6/breaking_updates_rl/pipeline/distillation/data1.csv"
    # read csv file into dict
    df = pd.read_csv(path_to_csv_info)
    newinfos = merge_infos(df, commit_hash_list)
    
    # save to csv
    output_path = "/home/xchen6/breaking_updates_rl/pipeline/distillation/data2.csv"
    newinfos.to_csv(output_path, index=False)
    
if __name__ == "__main__":
    main()
    