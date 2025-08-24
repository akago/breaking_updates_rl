import shutil
from pathlib import Path
import logging
from create_dataset import FailureCategoryExtract, FailureCategory

logger = logging.getLogger(__name__)

def move_compilation_failures(src_dir: str | Path, dst_dir: str | Path):
    """
    iterate over all .log files in src_dir
    move logs that belong to COMPILATION_FAILURE to dst_dir.
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    if not src_dir.exists():
        raise ValueError(f"src directory doesn't exist: {src_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)

    # count total .log files
    all_logs = list(src_dir.glob("*.log"))
    total_logs = len(all_logs)
    logger.info(f"Total log files found: {total_logs}")

    moved_count = 0
    for log_file in all_logs:
        extractor = FailureCategoryExtract(log_file)
        category = extractor.get_failure_category()
        if category == FailureCategory.COMPILATION_FAILURE:
            target_path = dst_dir / log_file.name
            # logger.info(f"move {log_file} -> {target_path}")
            shutil.move(str(log_file), str(target_path))
            moved_count += 1

    # final summary
    logger.info(f"Moved {moved_count} logs with category COMPILATION_FAILURE "
                f"out of {total_logs} total logs.")


# usecase
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    src_dir = Path(__file__).parent.parent / "external" / "bump_extension" / "reproductionLogs" / "successfulReproductionLogs"
    move_compilation_failures(src_dir, "compilation_failures")