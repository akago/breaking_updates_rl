#!/usr/bin/env python3
"""create_dataset.py

Clone the minimal history around a *breaking commit* plus grab the two
versions of the *updated dependency* described in a collection of JSON
files produced by the APR benchmark.

Usage
-----
    python create_dataset.py <json_dir> --output <output_dir>

Behaviour
---------
*   Only COMPILATION_FAILURE are considered.
*   For each such JSON descriptor we expect at least the fields

        projectOrganisation   (e.g. "davidmoten")
        project               (e.g. "rtree")
        breakingCommit        (40‑character SHA)
        updatedDependency     (object with dependencyGroupID, dependencyArtifactID,
                               previousVersion, newVersion)

*   The script creates ``<output_dir>/<SHA>/`` for every breaking commit and
    populates it with:

    ``repo/``   – A *shallow* clone that only contains the breaking commit
                  and its direct parent (``git fetch --depth 2``).

    ``deps/``   – Two Maven JARs corresponding to *previousVersion* and
                  *newVersion* of the updated dependency (downloaded from
                  Maven Central).

The clone is completely blob‑filtered and depth‑limited to 2 commits in order
to minimise disk usage.

Dependencies
------------
*   Python ≥ 3.8 (only standard library modules are used)
*   ``git`` must be available on the ``PATH``.

Notes
-----
*   If a repository has already been cloned the operation is skipped.
*   If a dependency JAR already exists it is not re‑downloaded.
*   Any failures (e.g., network outages, missing Maven artefacts) are logged
    and the script proceeds with the remaining work.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict




class BreakingDataset():
    def __init__(self, path_to_benchmark:str, outroot:str):
        self.outroot = outroot
        self.path_to_benchmark = path_to_benchmark
        pass
    
    def run(cmd: list[str], cwd: Path | None = None) -> None:
    """Run *cmd* (list of strings) and raise if the command fails."""
        print("$", *cmd, sep=" ")
        try:
            subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Command {' '.join(cmd)} failed with exit code {exc.returncode}") from exc
    
    
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


    def clone_repo(self, desc: Dict[str, Any]):
        sha = desc.get("breakingCommit")
        organisation = desc.get("projectOrganisation")
        project = desc.get("project")

        if not (sha and organisation and project):
            print("[warning] Missing mandatory fields; skipping entry")
            return

        repo_url = f"https://github.com/{organisation}/{project}.git"
        commit_dir = out_root / sha
        repo_dir = commit_dir / "repo"

        if not repo_dir.exists():
            print(f"Cloning {repo_url} at {sha} → {repo_dir}")
            # Create target directory
            repo_dir.parent.mkdir(parents=True, exist_ok=True)

            # Bare‑bones clone (no blobs, no checkout, shallow)
            run([
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
            run([
                "git",
                "-C",
                str(repo_dir),
                "fetch",
                "--depth",
                "2",
                "origin",
                sha,
            ])
            run(["git", "-C", str(repo_dir), "checkout", sha])
        else:
            print(f"[skipped] Repo for {sha} already exists")
    
    def download_jars(self, desc):
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
        dep_dir = commit_dir / "deps"

        for version in (prev_version, new_version):
            jar_url = f"{maven_base}/{group_path}/{artifact_id}/{version}/{artifact_id}-{version}.jar"
            jar_dest = dep_dir / f"{artifact_id}-{version}.jar"
            download(jar_url, jar_dest)
            
    def process_descriptor(self, desc: Dict[str, Any], out_root: Path) -> None:
        # filter compliation failures only
        if desc.get("failureCategory") != "COMPILATION_FAILURE":
            return

        # Clone repository (shallow, depth 2)
        self.clone_repo(desc)

        # Download dependency JARs (if any)
        self.download_jars(desc)

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download shallow clones and dependency JARs for APR descriptors.")
    parser.add_argument("json_dir", type=Path, help="Directory containing metadata JSON files for benchmarks")
    parser.add_argument("--output", "-o", type=Path, default=Path("output"), help="Output directory (default: ./output)")

    
    args = parser.parse_args(argv)

    
    json_files = sorted(Path(args.json_dir).glob("*.json"))
    if not json_files:
        print("No JSON files found in", args.json_dir)
        sys.exit(0)

    for json_file in json_files:
        try:
            with json_file.open("r", encoding="utf-8") as fp:
                descriptor = json.load(fp)
        except json.JSONDecodeError as exc:
            print(f"[warning] Could not parse {json_file}: {exc}")
            continue

        try:
            process_descriptor(descriptor, args.output)
        except Exception as exc:  # noqa: BLE001  (top‑level loop must catch‑all)
            print(f"[error] Failed to handle {json_file}: {exc}")


if __name__ == "__main__":
    main()
