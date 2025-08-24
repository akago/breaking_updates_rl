#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import re
import subprocess
import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple
import shutil
import logging
from enum import Enum
import pandas as pd

logger = logging.getLogger(__name__)

#=====================================CONSTATNTS=========================================#
# The failure classification follows the implementation in bacardi
class FailureCategory(Enum):
    JAVA_VERSION_FAILURE = "JAVA_VERSION_FAILURE"
    TEST_FAILURE = "TEST_FAILURE"
    WERROR_FAILURE = "WERROR_FAILURE"
    COMPILATION_FAILURE = "COMPILATION_FAILURE"
    BUILD_SUCCESS = "BUILD_SUCCESS"
    ENFORCER_FAILURE = "ENFORCER_FAILURE"
    DEPENDENCY_RESOLUTION_FAILURE = "DEPENDENCY_RESOLUTION_FAILURE"
    DEPENDENCY_LOCK_FAILURE = "DEPENDENCY_LOCK_FAILURE"
    UNKNOWN_FAILURE = "UNKNOWN_FAILURE"

# Failure patterns for failure classification
FAILURE_PATTERNS: Iterable[Tuple[re.Pattern, FailureCategory]] = [
    (re.compile(r"(?i)(class file has wrong version (\d+\.\d+), should be (\d+\.\d+))"), 
     FailureCategory.JAVA_VERSION_FAILURE),

    (re.compile(
        r"(?i)(\[ERROR\] Tests run:|There are test failures|There were test failures|"
        r"Failed to execute goal org\.apache\.maven\.plugins:maven-surefire-plugin)"
    ), FailureCategory.TEST_FAILURE),

    (re.compile(r"(?i)(warnings found and -Werror specified)"),
     FailureCategory.WERROR_FAILURE),

    (re.compile(
        r"(?i)(COMPILATION ERROR|Failed to execute goal io\.takari\.maven\.plugins:takari-lifecycle-plugin.*?:compile)"
        r"|Exit code: COMPILATION_ERROR"
    ), FailureCategory.COMPILATION_FAILURE),

    (re.compile(r"(?i)(BUILD SUCCESS)"),
     FailureCategory.BUILD_SUCCESS),

    (re.compile(
        r"(?i)(Failed to execute goal org\.apache\.maven\.plugins:maven-enforcer-plugin|"
        r"Failed to execute goal org\.jenkins-ci\.tools:maven-hpi-plugin)"
    ), FailureCategory.ENFORCER_FAILURE),

    (re.compile(
        r"(?i)(Could not resolve dependencies|\[ERROR\] Some problems were encountered while processing the POMs|"
        r"\[ERROR\] .*?The following artifacts could not be resolved)"
    ), FailureCategory.DEPENDENCY_RESOLUTION_FAILURE),

    (re.compile(
        r"(?i)(Failed to execute goal se\.vandmo:dependency-lock-maven-plugin:.*?:check)"
    ), FailureCategory.DEPENDENCY_LOCK_FAILURE),
]

# Dockerfile template for building containers
DEF_TEMPLATE = """Bootstrap: docker
From: ghcr.io/chains-project/breaking-updates:base-image

%post
    apk add fakeroot
    FAKEROOTDONTTRYCHOWN=1 fakeroot sh -c 'apk add openssh'
    git clone https://github.com/{organisation}/{project}.git
    cd {project}
    git fetch --depth 2 origin {commit_hash}

%environment
    export JAVA_HOME=/usr/lib/jvm/java-11-openjdk
    export MAVEN_HOME=/usr/share/maven
    export PATH=$JAVA_HOME/bin:$MAVEN_HOME/bin:$PATH

%runscript
    exec /bin/bash
    """
#=====================================CONSTATNTS=========================================#    


class FailureCategoryExtract:
    def __init__(self, log_file: Path | str):
        self.log_file = Path(log_file)

    def get_failure_category(self, log_file_path: Path | str | None = None) -> FailureCategory:
        """
        if log_file_path is provided, use it;
        otherwise fall back to the instance's self.log_file.
        """
        target = Path(log_file_path) if log_file_path is not None else self.log_file
        
        try:
            log_content = target.read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            logger.error("Failed to read log file: %s", str(target.resolve()))
            raise RuntimeError(f"Failed to read log file: {target}") from e

        for pattern, category in FAILURE_PATTERNS:
            if pattern.search(log_content):
                return category

        return FailureCategory.UNKNOWN_FAILURE

class BreakingDataset():
    def __init__(self, input:str, outroot:str):
        self.outroot = Path(outroot)
        self.current_dir = Path(input)
        self.path_to_benchmark = self.current_dir / "benchmark"
        self.path_to_build_logs = self.current_dir / "successfulReproductionLogs"
        self.dataset = []
        
    def build_dataset(self):
        """Build the dataset from the collected data."""
        # Ensure output directory exists
        self.outroot.mkdir(parents=True, exist_ok=True)

        # Iterate over all JSON files in the input directory
        for json_file in self.path_to_benchmark.glob("*.json"):
            with open(json_file, "r", encoding="utf-8") as f:
                desc = json.load(f)
                if not isinstance(desc, dict):
                    print(f"[warning] Invalid JSON format in {json_file}; skipping")
                    continue

                # Process each breaking update entry
                try:                
                    data_dict = self.process_breaking_update(desc, self.outroot)
                    if data_dict:
                        self.dataset.append(data_dict) 
                except Exception as exc:
                    print(f"[error] Failed to handle {json_file}: {exc}")
        print(f"Dataset built successfully at {self.outroot}")
        
    def run(self, cmd: list[str], cwd: Path | None = None) -> str:
        """Run *cmd* (list of strings) and raise if the command fails."""
        print("$", *cmd)
        cp = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,      # capture stdout/stderr
            text=True,                # decode output into string
            check=False               # handle the errors manually
        )
        if cp.returncode:
            # if failed
            raise RuntimeError(
                f"Command {' '.join(cmd)} failed (exit {cp.returncode}):\n"
                f"{cp.stderr}"
            )
        return cp.stdout
    
    
    def download(self, url: str, dest: Path) -> None:
        """Download *url* to *dest*, creating parent directories as needed."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            print(f"[skipped] {dest.name} already exists")
            return
        print(f"Downloading {url} -> {dest}")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as exc:  # noqa: BLE001  (broad exception is fine for top‑level tooling)
            print(f"[warning] Failed to download {url}: {exc}")

    def build_container(self, desc: Dict[str, Any], commit_dir: Path):
        
        project = desc.get("project")
        breakingCommit = desc.get("breakingCommit")
        organisation = desc.get("projectOrganisation")

        def_file = commit_dir / f"{project}.def"
        sif_file = commit_dir / f"{project}.sif"

        # generate .def file for each project
        with open(def_file, "w") as f:
            f.write(DEF_TEMPLATE.format(project=project, organisation=organisation, commit_hash=breakingCommit))

        logger.info(f"[INFO] Building container for {project} ...")

        # call apptainer build
        self.run([
            "apptainer", "build",
            "--fakeroot", "--ignore-fakeroot-command",
            str(sif_file),
            str(def_file)
        ])

        print(f"[DONE] Container built: {sif_file}")
        
    def clone_repo(self, desc: Dict[str, Any], commit_dir: Path):
        sha = desc.get("breakingCommit")
        organisation = desc.get("projectOrganisation")
        project = desc.get("project")

        if not (sha and organisation and project):
            print("[warning] Missing mandatory fields; skipping entry")
            return

        repo_url = f"https://github.com/{organisation}/{project}.git"
        
        repo_dir = commit_dir / project

        if not repo_dir.exists():
            print(f"Cloning {repo_url} at {sha} → {repo_dir}")
            # Create target directory
            repo_dir.parent.mkdir(parents=True, exist_ok=True)

            # Bare‑bones clone (no blobs, no checkout, shallow)
            self.run([
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                "--depth",
                "1",
                repo_url,
                str(repo_dir),
            ])

            # Fetch the breaking commit and its parent (depth 2) and check out.
            self.run([
                "git",
                "-C",
                str(repo_dir),
                "fetch",
                "--depth",
                "2",
                "origin",
                sha,
            ])
            self.run(["git", "-C", str(repo_dir), "checkout", sha])
        else:
            print(f"[skipped] Repo for {sha} already exists")

    def download_jars(self, desc: Dict[str, Any], commit_dir: Path):
        # download the previous and new verison jars of the dependency, then identify the breaking changes with roseau
        updated_dep = desc.get("updatedDependency")
        if not updated_dep:
            return

        group_id = updated_dep.get("dependencyGroupID")
        artifact_id = updated_dep.get("dependencyArtifactID")
        prev_version = updated_dep.get("previousVersion")
        new_version = updated_dep.get("newVersion")

        if not all((group_id, artifact_id, prev_version, new_version)):
            print("[warning] Incomplete dependency information; skipping JAR download")
            return

        maven_base = "https://repo1.maven.org/maven2"
        group_path = group_id.replace(".", "/")
        dep_dir = commit_dir

        # download previous and new verison jars
        for version in (prev_version, new_version):
            jar_url = f"{maven_base}/{group_path}/{artifact_id}/{version}/{artifact_id}-{version}.jar"
            jar_dest = dep_dir / f"{artifact_id}-{version}.jar"
            self.download(jar_url, jar_dest)

        # Get breaking changes between previous and newer version
        report_path = commit_dir / "api_diff.csv"
        breaking_changes = self.run([
            "java", 
            "-jar", 
            str(self.current_dir.parent / "external/roseau/cli/target/roseau-cli-0.2.0-SNAPSHOT-jar-with-dependencies.jar"), 
            "--diff",
            "--v1",
            str(dep_dir / f"{artifact_id}-{prev_version}.jar"),
            "--v2",
            str(dep_dir / f"{artifact_id}-{new_version}.jar"),
            "--format",
            "CSV",
            "--report",
            str(report_path),
            ])
        
        # read the diff into a dataframe
        all_cols = pd.read_csv(report_path, nrows=0).columns  # read the header only
        keep_cols = [c for c in all_cols if c in ('element', 'kind')]
        df = pd.read_csv(report_path, usecols=keep_cols, encoding='utf-8')
            
        return breaking_changes
        
        
    def process_breaking_update(self, desc: Dict[str, Any], out_root: Path) -> None:
        data_dict = {}

        # filter compilation failures only
        if desc.get("failureCategory") != "COMPILATION_FAILURE":
            return None
        commit_dir = self.outroot / Path(str(desc.get("breakingCommit")))
        
        # filter out the Java version failures
        build_log_path = self.path_to_build_logs / (str(desc.get('breakingCommit')) + ".log")
        extractor = FailureCategoryExtract(build_log_path) 
        failure_category = extractor.get_failure_category()
        print(f"[info] Failure category for {desc.get('breakingCommit')}: {failure_category}")
        if failure_category != FailureCategory.COMPILATION_FAILURE:
            print(f"[warning] Skipping {desc.get('breakingCommit')} due to java version failure")
            return None
        
        # Clone repository (shallow, depth 2)
        self.clone_repo(desc, commit_dir)

        # Build container
        self.build_container(desc, commit_dir)

        # Download dependency JARs (if any)
        breaking_changes = self.download_jars(desc, commit_dir)
        data_dict["breaking_changes"]= breaking_changes
        with open(commit_dir / "context.json", "w") as f:
            json.dump(data_dict, f)
        
        # Copy the build log to the project directory
        build_log_dest = commit_dir / desc.get("project") / (str(desc.get('breakingCommit')) + ".log")
        if build_log_path.exists():
            build_log_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(build_log_path, build_log_dest)
        else:
            print(f"[warning] Build log {build_log_path} does not exist; skipping copy")
            
        # Identify the error messages from the build log
        
def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Download shallow clones and dependency JARs for APR descriptors.")
    parser.add_argument("--input", "-i",  type=Path, default=Path(__file__).parent.parent/"data", help="Directory containing metadata JSON files for benchmarks & build logs. default: .")
    parser.add_argument("--output", "-o", type=Path, default=Path(__file__).parent.parent/"data"/"output1", help="Output directory (default: ./output)")
    args = parser.parse_args(argv)

    breaking_dataset = BreakingDataset(args.input, args.output)
    breaking_dataset.build_dataset()
    
    
if __name__ == "__main__":
    main()
