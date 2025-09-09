from pipeline.types.detected_files import DetectedFileWithErrors

class ProjectEnv:
    def __init__(self, init_files:List[DetectedFileWithErrors]=None):  # initial content of the buggy files
        self.init_files = init_files or []
        self.current_files = self.init_files.copy()
        self.num_files = len(self.init_files)
        self.num_errors = sum(self.count_errors(f.) for f in self.init_files)
        self.reset()

    def reset(self):
        """
        Reset the environment to the initial state.
        """
        self.
        return self.state()

    def count_errors(self, file_content):
        
        return file_content.count("ERROR")


    def state(self):
        per = [e / max(1, ie) for e, ie in zip(self.errors, self.init_files)]
        total = sum(self.errors) / max(1, sum(self.init_files))
        return np.array(per + [total], dtype=np.float32)

    def pick_file(self):
        cand = [i for i, e in enumerate(self.errors) if e > 0]
        return random.choice(cand) if cand else None

    def step_file(self, fidx: int, action: int):
        """
        action ∈ {0,1,2}:
          0 = invalid patch; 1 = fix partial error; 2 = fix all remaining errors
        local reward r_local = (fixed errors) / (errors before fixing)
        """
        before = self.errors[fidx]
        delta = 0 if action == 0 else (1 if action == 1 else before)
        after = max(0, before - delta)
        self.errors[fidx] = after
        r_local = (before - after) / max(1, before)   # ∈ [0,1]
        done = (sum(self.errors) == 0)                # if all files are fixed
        return self.state(), r_local, done
