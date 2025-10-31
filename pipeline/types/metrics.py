import subprocess
from pipeline.types.maven_error import MavenErrorLog, MavenErrorParser
from pipeline.types.utils import get_error_uid
# create java file with patch, binding with corresponding file in the container, compile with mvn test -DskipTests -B > breaking_commits
import logging

from pathlib import Path
from tempfile import TemporaryDirectory

class Patcher:
    def __init__(self, project:str, container_path:str, log_path: str=None,  binding_pairs:list[tuple[str,str]]=[]):
        self.container_path = container_path
        self.binding_pairs = binding_pairs # [(patch code, original code), ...]
        self.log_path = log_path
        self.project = project

    # to be fixed
    def apply_patch(self) -> tuple[dict, bool]:
        with TemporaryDirectory(prefix="job-", suffix="-ol") as jobdir:
            overlay_dir = Path(jobdir) / "upper"    
            overlay_dir.mkdir(parents=True, exist_ok=True)
            # basenames = Path(container_file).name
            # host_file = Path(jobdir) / basename
            # host_file.write_text(patch)
            
            # bind the patch files to the corresponding original files in the container
            bind_cmds_files = map(lambda x: f"{x[0]}:{x[1]}:ro", self.binding_pairs) # read-only bindings
            bind_cmds = ",".join(bind_cmds_files) 
            
            # add ca certificates binding
            # to avoid java ssl errors, add soft link the cacerts file in the container as well:  ln -sf /etc/pki/java/cacerts "$JAVA_HOME/lib/security/cacerts"
            bind_cmds += ",/etc/pki:/etc/pki:ro,/etc/ssl:/etc/ssl:ro"
            
            # replace the buggy file with patches by binding options in apptainer and run the tests
            cmd = ["apptainer", "exec", 
                "--pwd", f"/{self.project}", "--overlay", str(overlay_dir),
                "-B", bind_cmds, self.container_path, 
                "mvn", "-B", "-DskipTests", "clean", "test"]
            buf = []  
            logging.info(f"Running command: {' '.join(cmd)}")
            with open(self.log_path, "w", encoding="utf-8") as log_file:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,         
                    bufsize=1        
                )
                
                for line in proc.stdout:
                    log_file.write(line)
                    buf.append(line)
                proc.stdout.close()
                return_code = proc.wait()

            
            # analyze the log and get maven errors
            log_parser = MavenErrorParser()
            error_log = MavenErrorLog.from_string("".join(buf), log_parser)
            
            success = (return_code == 0)

            return error_log, success
    
    def apply_patch_training(self, patch:str, container_file:str) -> tuple[str, bool]:
        """apply a patch for a single file during training"""
        # create temporary directory for overlay fs
        with TemporaryDirectory(prefix="job-", suffix="-ol") as jobdir:
            overlay_dir = Path(jobdir) / "upper"    
            overlay_dir.mkdir(parents=True, exist_ok=True)
            basename = Path(container_file).name
            host_file = Path(jobdir) / basename
            host_file.write_text(patch)

            bind_cmds = f"{str(host_file)}:{container_file}:ro,"
            # Add ca certificates binding. To avoid java ssl errors, add soft link the cacerts file in the container as well:  ln -sf /etc/pki/java/cacerts "$JAVA_HOME/lib/security/cacerts"
            bind_cmds += "/etc/pki:/etc/pki:ro,/etc/ssl:/etc/ssl:ro"          
              
            # replace the buggy file with patches by binding options in apptainer and run the build
            cmd = ["apptainer", "exec", 
                "--pwd", f"/{self.project}", "--overlay", str(overlay_dir),
                "-B", bind_cmds, self.container_path, 
                "mvn", 
                # "-Dmaven.repo.local=" + str(Path(jobdir)/".m2repo"), 
                "-B", "-DskipTests", "clean", "test"]
            buf = []  
            logging.info(f"Running command: {' '.join(cmd)}")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,         
                bufsize=1        
            )
            for line in proc.stdout:
                # not need to write into disk
                buf.append(line)
            proc.stdout.close()
            return_code = proc.wait()

            # print("".join(buf))
            # build log, success status
            success = (return_code == 0)
            return "".join(buf), success
        
    def apply_patch_training_test(self, patch:str, container_file:str) -> tuple[str, bool]:
        """apply a patch for a single file during training"""
        # create temporary directory for overlay fs
        with TemporaryDirectory(prefix="job-", suffix="-ol") as jobdir:
            overlay_dir = Path(jobdir) / "upper"    
            overlay_dir.mkdir(parents=True, exist_ok=True)
            basename = Path(container_file).name
            host_file = Path(jobdir) / basename
            host_file.write_text(patch)

            bind_cmds = f"{str(host_file)}:{container_file}:ro,"
            # Add ca certificates binding. To avoid java ssl errors, add soft link the cacerts file in the container as well:  ln -sf /etc/pki/java/cacerts "$JAVA_HOME/lib/security/cacerts"
            bind_cmds += "/etc/pki:/etc/pki:ro,/etc/ssl:/etc/ssl:ro"          
              
            # replace the buggy file with patches by binding options in apptainer and run the build
            cmd = ["apptainer", "exec", 
                "--pwd", f"/{self.project}", "--overlay", str(overlay_dir),
                "-B", bind_cmds, self.container_path, 
                "mvn", 
                # "-Dmaven.repo.local=" + str(Path(jobdir)/".m2repo"), 
                "-B", "clean", "test"]
            buf = []  
            logging.info(f"Running command: {' '.join(cmd)}")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,         
                bufsize=1        
            )
            for line in proc.stdout:
                # not need to write into disk
                buf.append(line)
            proc.stdout.close()
            return_code = proc.wait()

            # print("".join(buf))
            # build log, success status
            success = (return_code == 0)
            return "".join(buf), success
    
    
    def get_metrics(self, original_errors: dict, current_errors: dict):
        """
        Given old and new errors in a buggy file, count the number of old errors, fixed errors and newly occurred errors in this file.
        """
        (file_name, error_msgs), = original_errors.items()
        original_error_count = len(error_msgs)
                
        original_error_uids = set(e["uid"] for e in error_msgs)
        # if the file is fully fixed
        if not file_name in current_errors:
            fixed_error_count = original_error_count
            new_errors_count = 0
        else:
            current_error_uids = set(get_error_uid(e["message"], e["additional_info"]) for e in current_errors[file_name])
            logging.info(f"Original errors uid in {file_name}: {original_error_uids}")
            logging.info(f"Current errors uid in {file_name}: {current_error_uids}\n")
            fixed_error_uids = original_error_uids - current_error_uids
            logging.info(f"File {file_name} fixed errors uid: {fixed_error_uids}\n")
            fixed_error_count = len(fixed_error_uids)
            new_errors_count = len(current_error_uids - original_error_uids)
        return original_error_count, fixed_error_count, new_errors_count
        
    def reward_dense(self, original_errors: dict, current_errors: dict, success: bool) -> float:
        # orinigal errors; dict: {filname: [error1, error2]}
        # current_errors; dict: [filname1: [error1, error2], file]
        eta = 0.2
        lmbda = 0.5
        
        # successful compilation
        if success and not current_errors:
            return 1.0
        
        # single file expected
        (file_name, error_msgs), = original_errors.items()
        original_error_count = float(len(error_msgs))
                
        original_error_uids = set(e["uid"] for e in error_msgs)
        # if the file is fully fixed
        if not file_name in current_errors:
            fixed_error_count = original_error_count
            return 1.0
        else:
            current_error_uids = set(get_error_uid(e["message"], e["additional_info"]) for e in current_errors[file_name])
            # logging.info(f"Original errors uid in {file_name}: {original_error_uids}")
            # logging.info(f"Current errors uid in {file_name}: {current_error_uids}\n")

            fixed_error_uids = original_error_uids - current_error_uids
            # logging.info(f"File {file_name} fixed errors uid: {fixed_error_uids}\n")
            
            fixed_error_count = float(len(fixed_error_uids))
            new_errors_flag = 1.0 if len(current_error_uids - original_error_uids) > 0 else 0.0
        
        return fixed_error_count / original_error_count - new_errors_flag * eta 
            
        
        
    
    @staticmethod
    def metrics(original_errors: dict, errors: dict, success: bool) -> dict:
        # dict: [filename1: [error1, error2, ...], filename2: [...], ...]
        # compare the result after applying the patch with initial compilation errors.
        
        original_error_count = sum(len(v) for v in original_errors.values())
        original_error_file_count = len(original_errors)
        
        if success and not errors:
            return {
                "original_file_count": original_error_count,
                "original_error_count": original_error_file_count,
                "build_success": True,
                "fixed_file_count": original_error_file_count,
                "fixed_error_count": original_error_count,
                "new_errors_count": 0,
                "fixed_errors": {k: len(v) for k, v in original_errors.items()},
                "new_errors": {},
            }
            
        build_success = False
        fixed_error_count = 0
        fixed_file_count = 0
        new_errors_count = 0
        fixed_errors = {}
        new_errors = {}
        
        logging.info(f"Original errors files: {original_errors.keys()}\n")
        logging.info(f"Current error files: {errors.keys()}\n")
        for file_name, error_list in original_errors.items():
            if file_name not in errors:
                fixed_file_count += 1
                fixed_error_count += len(error_list)
                fixed_errors[file_name] = len(error_list)
            else:
                original_error_uids = set(e["uid"] for e in error_list)
                current_error_uids = set(get_error_uid(e["message"], e["additional_info"]) for e in errors[file_name])
                logging.info(f"Original errors uid in {file_name}: {original_error_uids}")
                logging.info(f"Current errors uid in {file_name}: {current_error_uids}\n")

                fixed_error_uids = original_error_uids - current_error_uids
                logging.info(f"File {file_name} fixed errors uid: {fixed_error_uids}\n")
                
                fixed_error_count += len(fixed_error_uids)
                fixed_errors[file_name] = len(fixed_error_uids)
                new_errors_count += len(current_error_uids - original_error_uids)
                new_errors[file_name] = len(current_error_uids - original_error_uids)
        
        # count new files with errors
        for file_name in errors:
            if file_name not in original_errors:
                new_errors_count += len(errors[file_name])
                new_errors[file_name] = len(errors[file_name])
        
        return {
            "original_file_count": original_error_file_count,
            "original_error_count": original_error_count,
            "build_success": build_success,
            "fixed_file_count": fixed_file_count,
            "fixed_error_count": fixed_error_count,
            "new_errors_count": new_errors_count,
            "fixed_errors": fixed_errors,
            "new_errors": new_errors,
        }
