from pipeline.types.metrics import Patcher
from unittest import TestCase


class TestPatcher(TestCase):
    
    def test_apply_patch(self):
        patcher = Patcher(
            project="example-project",
            log_path="test.log",
            container_path="example-container.sif",
            binding_pairs=[("local_patch.java", "/container/path/buggy_file.java")]
        )
        try:
            errors, success = patcher.apply_patch()
            self.assertIsInstance(errors, dict)
            self.assertIsInstance(success, bool)
        except subprocess.CalledProcessError as e:
            self.fail(f"apply_patch raised CalledProcessError unexpectedly: {e}")
    
    

if __name__ == "__main__":
    import unittest
    unittest.main()
    