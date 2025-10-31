
from pathlib import Path
from pipeline.types.maven_error import MavenErrorLog, MavenErrorParser

if __name__ == "__main__":
    root_path = Path(__file__).parent.parent
    log = root_path / "data/dataset/0e8625f492854a78c0e1ceff67b2abd7e081d42b/jadler/0e8625f492854a78c0e1ceff67b2abd7e081d42b.log"
    
    log_parser = MavenErrorParser()
    error_log = MavenErrorLog.from_file(log, log_parser)

    for client_file_path, errors in error_log._by_path.items():
        print(f"Errors for {client_file_path}:")
        for client_line_position, error_info in errors.items():
            print(f"  Line {error_info.line_num} in {error_info.path}: {error_info.message}")
            if error_info.additional:
                print(f"    Additional Info: {error_info.additional}")
            print(f"    File Name: {error_info.path.name}")