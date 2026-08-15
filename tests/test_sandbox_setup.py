from pathlib import Path
import tempfile
import unittest

from src.inspection.config import InspectionConfig
from src.inspection.setup import inspect_setup


class FakeProbe:
    def __init__(self, python_path, executables=(), modules=()):
        self.python_path = str(python_path)
        self.executables = set(executables)
        self.modules = set(modules)

    def executable(self, name):
        return f"/usr/local/bin/{name}" if name in self.executables else None

    def module(self, name):
        return name in self.modules

    def python(self):
        return (3, 13), self.python_path


class SetupInspectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.px4 = self.root / "PX4-Autopilot"
        self.plan = self.root / "plan.json"
        self.collection = self.root / "collection"

    def tearDown(self):
        self.temporary.cleanup()

    def config(self, profile):
        return InspectionConfig(
            self.root, self.plan, self.collection, self.px4, profile=profile,
        )

    def test_demo_is_ready_without_simulator_dependencies(self):
        probe = FakeProbe("/usr/bin/python3", executables={"git"})
        result = inspect_setup(self.config("demo"), probe)
        self.assertTrue(result.ready)
        self.assertEqual(result.required_ready_count, result.required_count)
        items = {item.item_id: item for item in result.items}
        self.assertEqual(items["px4_checkout"].status, "optional")
        self.assertFalse(items["gz"].required)

    def test_development_reports_each_missing_required_dependency(self):
        probe = FakeProbe("/usr/bin/python3", executables={"git"})
        result = inspect_setup(self.config("development"), probe)
        self.assertFalse(result.ready)
        missing = {item.item_id for item in result.items if item.status == "missing"}
        self.assertTrue({
            "project_environment", "px4_checkout", "px4_sitl", "gz",
            "mavsdk", "simulation_worlds",
        }.issubset(missing))

    def test_base_interpreter_is_not_mistaken_for_project_environment(self):
        python = self.root / ".venv/bin/python"
        python.parent.mkdir(parents=True)
        python.write_text("", encoding="utf-8")
        probe = FakeProbe("/opt/homebrew/bin/python3", executables={"git"})
        result = inspect_setup(self.config("development"), probe)
        environment = next(
            item for item in result.items if item.item_id == "project_environment"
        )
        self.assertEqual(environment.status, "missing")

    def test_development_is_ready_with_project_simulator_stack(self):
        python = self.root / ".venv/bin/python"
        python.parent.mkdir(parents=True)
        python.write_text("", encoding="utf-8")
        (self.px4 / "build/px4_sitl_default/bin").mkdir(parents=True)
        (self.px4 / "CMakeLists.txt").write_text("", encoding="utf-8")
        (self.px4 / "build/px4_sitl_default/bin/px4").write_text("", encoding="utf-8")
        worlds = self.root / "simulation/worlds"
        worlds.mkdir(parents=True)
        (worlds / "training.sdf").write_text("<sdf/>", encoding="utf-8")
        probe = FakeProbe(python, executables={"git", "gz"}, modules={"mavsdk"})
        result = inspect_setup(self.config("development"), probe)
        self.assertTrue(result.ready)
        self.assertEqual(result.next_step, "Open Operator and run one flight smoke.")

    def test_report_contains_only_copyable_actions_and_official_links(self):
        probe = FakeProbe("/usr/bin/python3", executables={"git"})
        result = inspect_setup(self.config("development"), probe).to_record()
        px4 = next(item for item in result["items"] if item["item_id"] == "px4_checkout")
        gazebo = next(item for item in result["items"] if item["item_id"] == "gz")
        self.assertTrue(px4["docs_url"].startswith("https://docs.px4.io/"))
        self.assertTrue(gazebo["docs_url"].startswith("https://gazebosim.org/"))
        self.assertNotIn("execute", result)


if __name__ == "__main__":
    unittest.main()
