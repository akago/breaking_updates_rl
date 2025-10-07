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
from typing import Any, Iterable, Tuple
import shutil
import logging
from enum import Enum
import pandas as pd
import shlex
import tempfile




from pipeline.types.project import Project 
from pipeline.types.maven_error import MavenErrorLog, MavenErrorParser, ErrorInfo

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

%files
    {repo_path} /{project}

%post
    apk add fakeroot
    FAKEROOTDONTTRYCHOWN=1 fakeroot sh -c 'apk add openssh'
    cd /{project}

%environment
    export JAVA_HOME=/usr/lib/jvm/java-11-openjdk
    export MAVEN_HOME=/usr/share/maven
    export PATH=$JAVA_HOME/bin:$MAVEN_HOME/bin:$PATH

%runscript
    exec /bin/bash
    """
    

DEF_TEMPLATE_DOWNLOAD = """Bootstrap: docker
From: ghcr.io/chains-project/breaking-updates:base-image

%post
    apk add fakeroot
    FAKEROOTDONTTRYCHOWN=1 fakeroot sh -c 'apk add openssh'
    git clone https://github.com/{organisation}/{project}.git
    cd /{project}
    git fetch --depth 1 origin {commit_hash}
    git checkout {commit_hash}

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
        self.input_dir = Path(input)
        self.path_to_benchmark = self.input_dir / "benchmark"
        self.path_to_build_logs = self.input_dir / "successfulReproductionLogs"
        self.current_dir = Path(__file__).parent
        self.dataset = []

    def build_dataset(self) -> None:
        """Build the dataset from the collected data."""
        # Ensure output directory exists
        self.outroot.mkdir(parents=True, exist_ok=True)
        success, failure, fqcn_failure, bc_failure = [], [], [], []
        # Iterate over all JSON files in the input directory
        for json_file in self.path_to_benchmark.glob("*.json"):
            filename = json_file.stem
            with open(json_file, "r", encoding="utf-8") as f:
                desc = json.load(f)
                if not isinstance(desc, dict):
                    logger.warning(f"[warning] Invalid JSON format in {json_file}; skipping")
                    continue
                # filter out the Java version failures
                build_log_path = self.path_to_build_logs / (str(desc.get('breakingCommit')) + ".log")
                extractor = FailureCategoryExtract(build_log_path) 
                failure_category = extractor.get_failure_category()
                logger.info(f"Failure category for {desc.get('breakingCommit')}: {failure_category}")
                if failure_category != FailureCategory.COMPILATION_FAILURE:
                    # move the failures to another folder
                    failure_folder = self.outroot.parent / "failedGeneration"
                    commit_dir = self.outroot / desc.get('breakingCommit')
                    if commit_dir.exists():
                        cmd = [
                            "mv",
                            str(commit_dir),
                            str(failure_folder),
                        ]
                        self.run(cmd)
                    logger.warning(f"Skipping {desc.get('breakingCommit')} due to java version failure")
                    continue

                # Process each breaking update entry
                try:                
                    self.process_breaking_update(desc, self.outroot)
                except RuntimeError as re:
                    if "No relevant breaking changes found" in str(re):
                        bc_failure.append(filename)
                    elif "No FQCNs extracted" in str(re):
                        fqcn_failure.append(filename)
                    else:
                        failure.append(filename)
                except Exception as exc:
                    logger.error(f"Failed to handle {json_file}: {exc}")
                    failure.append(filename)
                
                success.append(filename)
                
        json.dump({"success_count":len(success),"failure_count":len(failure), 
                   "bc_failure_count":len(bc_failure), "fqcn_failure_count":len(fqcn_failure),
                   "failure": failure, "bc_failure": bc_failure, 
                   "fqcn_failure": fqcn_failure, "success": success
                   }, 
                  open("dataset_metadata.json", "w"), 
                  indent=4)
        logger.info(f"Dataset built successfully at {self.outroot}")

    def run(self, cmd: list[str], cwd: Path | None=None, capture_output: bool = True) -> str:
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
    
    
    def download(self, url: str, dest: Path) -> None:
        """Download *url* to *dest*, creating parent directories as needed."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            logger.info(f"[skipped] {dest.name} already exists")
            return
        logger.info(f"Downloading {url} -> {dest}")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as exc:  # noqa: BLE001  (broad exception is fine for top‑level tooling)
            logger.error(f"[error] Failed to download {url}: {exc}")

    
    
    
    def build_container_download(self, desc: dict[str, Any], commit_dir: Path) -> None:
        
        project = desc.get("project")
        organisation = desc.get("projectOrganisation")
        commit_hash = desc.get("breakingCommit")
        
        def_file = commit_dir / f"{project}.def"
        sif_file = commit_dir / f"{project}.sif"

        # if .sif already exists
        if sif_file.exists():
            return None
        # generate .def file for each project
        with open(def_file, "w") as f:
            f.write(DEF_TEMPLATE_DOWNLOAD.format(project=project, organisation=organisation, commit_hash=commit_hash))

        logger.info(f"[INFO] Building container for {project} ...")

        # call apptainer build
        self.run([
            "apptainer", "build",
            "--fakeroot", "--ignore-fakeroot-command",
            str(sif_file),
            str(def_file)
        ])

        logger.info(f"[DONE] Container built: {sif_file}")
        
    def build_container(self, desc: dict[str, Any], commit_dir: Path) -> None:
        
        project = desc.get("project")
        
        def_file = commit_dir / f"{project}.def"
        sif_file = commit_dir / f"{project}.sif"

        # if .sif already exists
        if sif_file.exists():
            return None
        
        # generate .def file for each project
        with open(def_file, "w") as f:
            f.write(DEF_TEMPLATE.format(project=project, repo_path= str(commit_dir / project)))

        logger.info(f"[INFO] Building container for {project} ...")

        # call apptainer build
        self.run([
            "apptainer", "build",
            "--fakeroot", "--ignore-fakeroot-command",
            str(sif_file),
            str(def_file)
        ])

        logger.info(f"[DONE] Container built: {sif_file}")
        
    def clone_repo(self, desc: dict[str, Any], commit_dir: Path) -> None:
        sha = desc.get("breakingCommit")
        organisation = desc.get("projectOrganisation")
        project = desc.get("project")

        if not (sha and organisation and project):
            logger.warning("Missing mandatory fields; skipping entry")
            return

        repo_url = f"https://github.com/{organisation}/{project}.git"
        
        repo_dir = commit_dir / project

        if not repo_dir.exists():
            logger.info(f"Cloning {repo_url} at {sha} → {repo_dir}")
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
            logger.info(f"[skipped] Repo for {sha} already exists")


    def snapshot_repo(self, desc: dict, commit_dir: Path) -> None:
        """
        Create a shallow clone of the repository at the specified commit.
        This avoids cloning the entire repository history.
        """
        sha = desc.get("breakingCommit")
        org = desc.get("projectOrganisation")
        project = desc.get("project")
        if not (sha and org and project):
            logger.warning("Missing mandatory fields; skipping entry")
            return

        repo_url = f"https://github.com/{org}/{project}.git"
        repo_dir = commit_dir / project
        if repo_dir.exists():
            logger.info(f"[skipped] Repo for {sha} already exists at {repo_dir}")
            return

        repo_dir.parent.mkdir(parents=True, exist_ok=True)

        # shallow clone 
        self.run(["git", "init", str(repo_dir)])
        self.run(["git", "-C", str(repo_dir), "remote", "add", "origin", repo_url])
        self.run(["git", "-C", str(repo_dir), "fetch", "--depth", "1", "origin", sha])
        self.run(["git", "-C", str(repo_dir), "checkout", "--detach", "FETCH_HEAD"])
    
    
    def download_jars(self, desc: dict[str, Any], commit_dir: Path) -> None:
        """
        Download the previous and new version artifacts of the dependency.
        Fallback to Jenkins repo for for Jenkins plugins.
        """
        updated_dep = desc.get("updatedDependency")
        if not updated_dep:
            return

        group_id = updated_dep.get("dependencyGroupID")
        artifact_id = updated_dep.get("dependencyArtifactID")
        prev_version = updated_dep.get("previousVersion")
        new_version = updated_dep.get("newVersion")

        if not all((group_id, artifact_id, prev_version, new_version)):
            logger.warning("[warning] Incomplete dependency information; skipping JAR download")
            return
        
        group_path = group_id.replace(".", "/")
        dep_dir = commit_dir
        base = "https://repo1.maven.org/maven2" if group_id not in ["org.jenkins-ci.plugins", "org.jenkins-ci"] else "https://repo.jenkins-ci.org/releases"

        def try_download(version: str) -> None:
            url = f"{base}/{group_path}/{artifact_id}/{version}/{artifact_id}-{version}.jar"
            dst = dep_dir / f"{artifact_id}-{version}.jar"
            self.download(url, dst)
            logger.info(f"[ok] Downloaded: {url}")
               
        # previous & new
        try_download(prev_version)
        try_download(new_version)

        # Get breaking changes between previous and newer version
        report_path = commit_dir / "api_diff.json"
        
        breaking_changes = self.run([
            "java", 
            "-jar", 
            str(self.current_dir / "libs/roseau-cli-0.2.0/roseau-cli-0.2.0-SNAPSHOT-jar-with-dependencies.jar"), 
            "--diff",
            "--v1",
            str(dep_dir / f"{artifact_id}-{prev_version}.jar"),
            "--v2",
            str(dep_dir / f"{artifact_id}-{new_version}.jar"),
            "--format",
            "JSON",
            "--report",
            str(report_path),
            ],
                                    #  capture_output=False
                                     )

        
        
    def filter_breaking_changes(self, breaking_changes: list[dict], FQCNs: list[str]) -> list[dict]:
        """Filter breaking changes to include only those relevant to the client code."""
        changes = set()
        for change in breaking_changes:
            # Check if any FQCN or class name matches the changed element
            if any(fqcn in change["element"] or fqcn.split(".")[-1] == change["element"].split(".")[-1] for fqcn in FQCNs):
                change = {"element": change["element"], "nature": change["nature"], "kind": change["kind"]}
                simplified_change = (
                    change["element"],
                    change["nature"],
                    change["kind"],
                )
                changes.add(simplified_change)
        return [
            {"element": e, "nature": n, "kind": k}
            for (e, n, k) in changes
        ]

    def generate_contexts(self, desc: dict[str, Any], commit_dir: Path) -> None:
        context_path = commit_dir / "context.json" 
      
        context_json = {}
        # project info
        context_json["project"] = desc.get("project")
        context_json["breakingCommit"] = desc.get("breakingCommit")
        context_json["projectOrganisation"] = desc.get("projectOrganisation")
        context_json["libraryName"] = desc.get("updatedDependency").get("dependencyArtifactID")
        context_json["libraryGroupID"] = desc.get("updatedDependency").get("dependencyGroupID")
        context_json["previousVersion"] = desc.get("updatedDependency").get("previousVersion")
        context_json["newVersion"] = desc.get("updatedDependency").get("newVersion")

        # generate the context for each buggy file in project
        try:
            with open(str(commit_dir / "api_diff.json")) as f:
                breaking_changes = json.load(f)
        except Exception as e:
            raise e
            
        # Extract relevant FQCNs from the client code        
        client = commit_dir
        logger.info(f"Extracting FQCNs from client code at {client}")
        log = self.path_to_build_logs / (desc.get('breakingCommit') + ".log")
        cmd = ['java', '-jar', 'pipeline/libs/fqcn-extractor/target/FqcnExtractor.jar', '-c', str(client), '-l', str(log)]
        try:
            result = self.run(cmd).strip()
        except RuntimeError as e:
            logger.error(f"Failed to extract FQCNs: {e}")
            result = "[]"

        logger.info("Extracted FQCNs: %s", repr(result))
        FQCNs = json.loads(result)
        # No FQCNs extracted
        if not FQCNs:
            raise RuntimeError("No FQCNs extracted")
        
        # filter only the breaking changes that are potentially relevant to the compilation errors
        relevant_changes = self.filter_breaking_changes(breaking_changes, FQCNs)
        logger.info("Relevant BCs: %s", relevant_changes)
        if not relevant_changes:
            raise RuntimeError("No relevant breaking changes found")
        context_json["breakingChanges"] = relevant_changes
        context_json["FQCNs"] = FQCNs
        
        # get errors information from the build log
        log_parser = MavenErrorParser()
        try:
            error_log = MavenErrorLog.from_file(log, log_parser)
        except Exception as exc:
            logger.error(f"Failed to parse build log for {desc.get('breakingCommit')}: {exc}")
            raise exc

        context_json["buggyFiles"] = error_log.to_jsonable()
        
        with open(commit_dir / "context.json", "w") as f:
            json.dump(context_json, f, indent=4)

    def process_breaking_update(self, desc: dict[str, Any], out_root: Path) -> None:
        # filter compilation failures only
        if desc.get("failureCategory") != "COMPILATION_FAILURE":
            return None
        commit_dir = self.outroot / Path(str(desc.get("breakingCommit")))
        # commit_dir.mkdir(parents=True, exist_ok=True)
        
        # # Pull the repo snapshot at the breaking commit
        # # self.clone_repo(desc, commit_dir)
        # self.snapshot_repo(desc, commit_dir)

        # Build container
        
        # Download dependency JARs (if any) and identify breaking changes with roseau
        # breaking_changes = self.download_jars(desc, commit_dir)
        
        
        # # Copy the build log to the project directory
        # build_log_dest = commit_dir / desc.get("project") / (str(desc.get('breakingCommit')) + ".log")
        # if build_log_path.exists():
        #     build_log_dest.parent.mkdir(parents=True, exist_ok=True)
        #     shutil.copy2(build_log_path, build_log_dest)
        # else:
        #     logger.warning(f"Build log {build_log_path} does not exist; skipping copy")

        # Identify the error messages from the build log
        # try:
        #     self.generate_contexts(desc, commit_dir)
        #     # self.build_container_download(desc, commit_dir)
            
        # except Exception as exc:
        #     logger.error(f"Failed to generate contexts for {desc.get('breakingCommit')}: {exc}")
        #     # move the failures to another folder
        #     failure_folder = commit_dir.parent.parent / "failedGeneration"
        #     if commit_dir.exists():
        #         cmd = [
        #             "mv",
        #             str(commit_dir),
        #             str(failure_folder),
        #         ]
        #         self.run(cmd)
        #     raise exc
        
        

def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Download shallow clones and dependency JARs for APR descriptors.")
    parser.add_argument("--input", "-i",  type=Path, default=Path(__file__).parent.parent/"data", help="Directory containing metadata JSON files for benchmarks & build logs. default: data/")
    parser.add_argument("--output", "-o", type=Path, default=Path(__file__).parent.parent/"data"/"dataset", help="Output directory (default: data/output/)")
    args = parser.parse_args(argv)

    breaking_dataset = BreakingDataset(args.input, args.output)
    breaking_dataset.build_dataset()
    
    
    
    
if __name__ == "__main__":
    main()
