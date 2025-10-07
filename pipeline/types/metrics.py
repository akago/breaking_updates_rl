

# create java file with patch, binding with corresponding file in the container, compile with mvn test -DskipTests -B > breaking_commits

class Patcher:
    def __init__(self):
        self.container = 
        self.file_path =
        self.file_name 
        self.patch
        self.tmp_path
        
        
    def create_fixed_temp_file(self, tmp_file:str) -> None:
        with open(tmp_file, "w") as f:
            f.write(self.path)
             
        
    def apply_patch(patch:str=self.patch, file_path=):
        # create tmp file
        tmp_file = tmp_file = self.tmp_path / self.file_name
        self.create_fixed_temp_file(tmp_file)
        # path to new log 
        log_file = 
        with open(log_file, "w") as lf:
            cmd = ["apptainer", "exec", "-B", f"{str(tmp_file)}:{str(file_path)}", str(self.container), "mvn", "-B", "-DskipTests", "clean", "test"]
            subprocess.run(
                cmd,
                check=True,
                stdout=lf,
                stderr=subprocess.STDOUT
                )
        # analyze the log and get maven errors
        
        return errors, success
    

    def metrics(self, original_erorrs, errors):
        # compare the result after applying the patch with initial compilation errors.
        
    