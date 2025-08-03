


class Environment:
    def __init__(self, name, version):
        self.name = name
        self.version = version

    def __repr__(self):
        return f"Environment(name={self.name}, version={self.version})"

    def get_reward(self):
        # Placeholder for reward calculation logic
        return 0.0
    
    def compile_test_project(self):
        # Placeholder for project compilation logic
        print(f"Compiling project in {self.name} environment with version {self.version}")

    def 