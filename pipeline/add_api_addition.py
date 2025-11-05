
from pipeline.types.maven_error import MavenErrorParser, MavenErrorLog
from pathlib import Path
import subprocess
import json
import re

def run(cmd: list[str], cwd: Path | None=None, capture_output: bool = True) -> str:
        """Run *cmd* (list of strings) and raise if the command fails."""
        # logger.info("$", *cmd)
        
        cp = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=capture_output,      # capture stdout/stderr
            text=True,                # decode output into string
            check=False               # handle the errors manually
        )
        if cp.returncode:
            # if failed
            raise RuntimeError(
                f"Command {' '.join(cmd)} failed (exit {cp.returncode}):\n"
                f"{cp.stderr}"
            )
        # return standard output
        return cp.stdout

def filter_api_additions(api_additions: list[str], FQCNs: list[str], missing_symbols: list[str]) -> list[str]:
    """Filter breaking changes to include only those relevant to the client code."""
    additions = []
    for addition in api_additions:
        # Check if any FQCN or class name matches the API addition
        addition_fqcn = addition.split("(")[0].split(" ")[-1]
        if any(fqcn == addition_fqcn or fqcn.split(".")[-1] == addition_fqcn.split(".")[-1] for fqcn in FQCNs):
            additions.append(addition)
        if any(addition_fqcn.split(".")[-1] == symbol for symbol in missing_symbols):
            additions.append(addition)
    return additions



def filter_with_error_message_changes(breaking_changes: list[dict], buggyFiles: dict):
    """
    find if the last element in the fqcn contained by an error message
    """
    final_BCs = []
    for file_name, errors in buggyFiles.items():
        errors_in_file_with_BCs = []
        for error in errors:
            error_message = f"{error.get('message','')}{error.get('additional_info','')}"
            relevant_changes_to_error = [
                change for change in breaking_changes
                if change["element"] in error_message or change["element"].split(".")[-1] in error_message
            ]
            for change in relevant_changes_to_error:
                if change not in final_BCs:
                    final_BCs.append(change)
            error["BCs"] = relevant_changes_to_error
            errors_in_file_with_BCs.append(error)
            
        buggyFiles[file_name] = errors_in_file_with_BCs
    return buggyFiles, final_BCs

def filter_with_error_message_additions(additions: list[str], buggyFiles: dict):
    """
    find if the last element in the fqcn contained by an error message
    """
    final_additions = []
    for file_name, errors in buggyFiles.items():
        errors_in_file_with_BCs = []
        for error in errors:
            error_message = f"{error.get('message','')}{error.get('additional_info','')}"
            relevant_additions_to_error = [
                addition for addition in additions
                if (fqcn := addition.split("(")[0].split(" ")[-1]) and (
                    fqcn in error_message or fqcn.split(".")[-1] in error_message
                )
            ]
            final_additions.extend(relevant_additions_to_error)
            error["Additions"] = relevant_additions_to_error
            errors_in_file_with_BCs.append(error)
              
        buggyFiles[file_name] = errors_in_file_with_BCs
    return buggyFiles, list(set(final_additions))

def get_missing_symbols_in_errors(buggyFiles):
    def extract_symbol_names(s: str):
        pattern = r'(?m)^\s*symbol:\s*(?:method|variable|class|interface|enum|constructor)\s+([A-Za-z_$][\w$]*)'
        return list(dict.fromkeys(re.findall(pattern, s)))
    result = []
    for _, errors in buggyFiles.items():
        for error in errors:
            symbol = extract_symbol_names(f'{error["additional_info"]}')
            result.extend(symbol)
    return set(result)    

data_path = Path(__file__).parent.parent / "data" / "dataset"
    

for folder in data_path.iterdir():
    context = folder / "context.json"
    context_dict = json.loads(context.read_text())
    
    client = folder / f"{context_dict['project']}"
    old_jar = folder / f"{context_dict['libraryName']}-{context_dict['previousVersion']}.jar"
    new_jar = folder / f"{context_dict['libraryName']}-{context_dict['newVersion']}.jar"
    log = folder / f"{context_dict['project']}" / f"{context_dict['breakingCommit']}.log"
    breaking_changes = context_dict["breakingChanges"]
    buggyFiles = context_dict["buggyFiles"]
    library_group_id = context_dict["libraryGroupID"]
    # only need the fqcns of dependency library
    FQCNs = [x for x in context_dict["FQCNs"] if library_group_id in x or library_group_id.split(".")[-1] in x]
    # special case for jakarta
    if "jakarta" in library_group_id:
        FQCNs += [x for x in context_dict["FQCNs"] if "javax" in x]
    # get missing symbols
    missing_symbols = list(get_missing_symbols_in_errors(buggyFiles))
    
    # Filter API Additions
    print(f"Extracting API additions from client code at {context_dict['breakingCommit']}/{context_dict['project']}")
    api_addition_cmd = ['java', '-jar', 'pipeline/libs/api_additions_lister/api-additions-lister-1.0.0.jar', str(old_jar), str(new_jar)]
    api_addition_json = folder / "api_additions.json"
    
    if api_addition_json.exists():
        print(f"API addition file already exists! Reading from local disk")
        api_additions = json.loads(api_addition_json.read_text())
    else:
        try:
            api_addition_result = run(api_addition_cmd).strip()
            print(f"API addition extracted!")
            print(api_addition_result)
        except RuntimeError as e:
            print(f"Failed to extract API additions with japicmp: {e}")
            result = "[]"
        api_additions = api_addition_result.split("\n")
        # to disk
        with open(str(api_addition_json), "w") as f:
            json.dump(api_additions, f)    
    relevant_additions = filter_api_additions(api_additions, FQCNs, missing_symbols)
                    
    print("\nRelevant Additions: %s", relevant_additions)
    if not relevant_additions:
        print("\n[ERROR] No relevant Additions found")
    
    # Further Flitering of BCs and Additions
    print(f"Adding BCs to Errors...")
    buggyFileswithBCs, final_BCs = filter_with_error_message_changes(breaking_changes, buggyFiles)
    print(f"Adding Addtions to Errors...")
    buggyFileswithAdditions, final_additions = filter_with_error_message_additions(relevant_additions, buggyFileswithBCs)
    
    new_context_dict = context_dict
    new_context_dict["buggyFiles"] = buggyFileswithAdditions
    new_context_dict["apiAdditions"] = final_additions
    new_context_dict["breakingChanges"] = final_BCs
    
    with open(folder / "new_context.json", "w") as f:
        json.dump(new_context_dict, f, indent=4)
