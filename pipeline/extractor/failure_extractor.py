import json
import subprocess
from typing import List

from pipeline.types.failure import Failure
from pipeline.types.project import Project


class FailureExtractor:
    def __init__(self, project: Project):
        self.project = project

    def get_failures(self, base_path=None) -> List[Failure]:
        base_path = base_path or self.project.path
        log_path = f"{base_path}/{self.project.project_name}/{self.project.project_id}.log"

        old_dependency = f"{self.project.library_name}-{self.project.old_library_version}"
        new_dependency = f"{self.project.library_name}-{self.project.new_library_version}"

        old_dependency_path = f"{base_path}/{old_dependency}.jar"
        new_dependency_path = f"{base_path}/{new_dependency}.jar"

        result = subprocess.run([
            'java',
            '-jar', 'libs/java/target/Explaining.jar',
            '-c', base_path,
            '-o', old_dependency_path,
            '-n', new_dependency_path,
            '-l', log_path,
        ], stdout=subprocess.PIPE)
        json_data = json.loads(result.stdout)

        return [Failure.from_json(row) for row in json_data]

    
    def get_project(self) -> Project:
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
        with open(str(commit_dir / "api_diff.json")) as f:
            breaking_changes = json.load(f)
            
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
        
        return self.project