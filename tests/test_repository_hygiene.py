import subprocess
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
    def test_ai_agent_prompt_files_are_not_tracked(self):
        if not (PROJECT_ROOT / ".git").exists():
            self.skipTest("Git metadata is unavailable")
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


if __name__ == "__main__":
    unittest.main()
