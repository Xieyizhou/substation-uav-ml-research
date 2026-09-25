import subprocess
import json
import shutil
import tempfile
import unittest
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_BASENAMES = {
    "AGENTS.md",
    "CLAUDE.md",
    "CODEBUDDY.md",
    "GEMINI.md",
    "QODER.md",
    ".cursorrules",
    ".mcp.json",
    ".windsurfrules",
    "copilot-instructions.md",
    "opencode.json",
    "opencode.jsonc",
}
FORBIDDEN_DIRECTORY_PARTS = {
    ".agents",
    ".claude",
    ".codex",
    ".code-review-graph",
    ".codebuddy",
    ".continue",
    ".cursor",
    ".gemini",
    ".kiro",
    ".qoder",
    ".roo",
    ".windsurf",
}
FORBIDDEN_GITHUB_DIRECTORIES = {
    (".github", "instructions"),
    (".github", "prompts"),
}
FORBIDDEN_SUFFIXES = (
    ".instruction.md",
    ".instructions.md",
    ".prompt.md",
    ".prompt.txt",
)
FORBIDDEN_NAME_FRAGMENTS = (
    "agent-prompt",
    "agent_prompt",
    "system-prompt",
    "system_prompt",
)


def tracked_files():
    if not (PROJECT_ROOT / ".git").exists():
        manifest = json.loads((PROJECT_ROOT / "SOURCE_MANIFEST.json").read_text())
        return [PurePosixPath(row["path"]) for row in manifest]
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    )
    return [
        PurePosixPath(path.decode())
        for path in result.stdout.split(b"\0")
        if path
    ]


def is_agent_prompt_path(path):
    lowered_name = path.name.lower()
    lowered_parts = tuple(part.lower() for part in path.parts)
    if path.name in FORBIDDEN_BASENAMES:
        return True
    if FORBIDDEN_DIRECTORY_PARTS.intersection(lowered_parts):
        return True
    if any(
        directory == lowered_parts[: len(directory)]
        for directory in FORBIDDEN_GITHUB_DIRECTORIES
    ):
        return True
    if lowered_name.endswith(FORBIDDEN_SUFFIXES):
        return True
    return any(fragment in lowered_name for fragment in FORBIDDEN_NAME_FRAGMENTS)


class RepositoryHygieneTests(unittest.TestCase):
    def setUp(self):
        # Test ignore rules independently of local Git metadata. Source ZIPs
        # intentionally omit .git, and Git worktrees may use a pointer file.
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.ignore_root = Path(temporary.name)
        subprocess.run(["git", "init", "-q", str(self.ignore_root)], check=True)
        shutil.copy2(PROJECT_ROOT / ".gitignore", self.ignore_root / ".gitignore")

    def test_ai_agent_prompt_files_are_not_tracked(self):
        prohibited = sorted(
            str(path)
            for path in tracked_files()
            if is_agent_prompt_path(path)
        )
        self.assertEqual(
            prohibited,
            [],
            "AI agent prompt/instruction files must remain local and untracked",
        )

    def test_visual_payload_model_and_benchmark_outputs_are_ignored(self):
        generated = (
            "data/research/visual_pilot/frames/000000001.png",
            "data/research/visual_pilot/annotations.jsonl",
            "datasets/visual/dataset_identity.json",
            "models/equipment/candidate.pt",
            "models/equipment/candidate.onnx",
            "outputs/research/visual_static_v1/results.json",
            "outputs/research/visual_static_v1/runtime_environment.json",
        )
        result = subprocess.run(
            ["git", "check-ignore", "--stdin"],
            cwd=self.ignore_root,
            input="\n".join(generated),
            text=True,
            check=True,
            capture_output=True,
        )
        self.assertEqual(set(result.stdout.splitlines()), set(generated))

    def test_visual_schemas_and_benchmark_templates_are_not_ignored(self):
        tracked_definitions = (
            "config/schemas/visual_dataset_identity.schema.json",
            "config/schemas/visual_collection_protocol.schema.json",
            "config/schemas/visual_collection_plan.schema.json",
            "config/schemas/visual_training_view_identity.schema.json",
            "benchmarks/visual_static_v1/conditions.json",
            "benchmarks/visual_static_v1/pilot_protocol.json",
            "benchmarks/visual_static_v1/collection_protocol.json",
        )
        result = subprocess.run(
            ["git", "check-ignore", "--stdin"],
            cwd=self.ignore_root,
            input="\n".join(tracked_definitions),
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
