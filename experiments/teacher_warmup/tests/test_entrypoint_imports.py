"""Exercise real package initialization in a fresh process without physics libraries."""
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
BOOTSTRAP = r'''
import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

root, script, mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
sys.path.insert(0, str(root / "legged_gym"))

def stub(name, **attrs):
    module = ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module
    return module

# Replace only heavy dependencies. Keep the actual envs/utils initializers,
# task_registry and base configuration modules, where the import cycle occurs.
for name in ("isaacgym", "torch", "numpy"):
    stub(name)
stub("rsl_rl", __path__=[])
stub("rsl_rl.env", VecEnv=object)
stub("rsl_rl.runners", HIMOnPolicyRunner=object)
helper_names = ("export_policy_as_jit export_jit_to_onnx load_onnx_policy "
                "class_to_dict get_load_path set_seed update_class_from_dict "
                "update_cfg_from_args parse_sim_params").split()
helpers = stub("legged_gym.utils.helpers",
               **{name: (lambda *a, **kw: None) for name in helper_names})
helpers.get_args = lambda: SimpleNamespace(seed=None, num_envs=None)
stub("legged_gym.utils.logger", Logger=object)
stub("legged_gym.utils.math", __all__=[])
stub("legged_gym.utils.terrain", Terrain=object)

env_init = root / "legged_gym/legged_gym/envs/__init__.py"
for node in ast.parse(env_init.read_text(encoding="utf-8")).body:
    if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module.startswith("g1."):
        name = "legged_gym.envs." + node.module
        module = sys.modules.get(name) or stub(name)
        module.__dict__.update({alias.name: type(alias.name, (), {}) for alias in node.names})

path = root / "legged_gym/legged_gym/scripts" / (script + ".py")
spec = importlib.util.spec_from_file_location(script, path)
entrypoint = importlib.util.module_from_spec(spec)
source = path.read_text(encoding="utf-8")
if mode == "before_fix":
    source = source.replace('    __import__("legged_gym.envs")\n', '')
exec(compile(source, str(path), "exec"), entrypoint.__dict__)
sys.argv = [str(path), "--teacher_checkpoint", "teacher.pt",
            "--student_checkpoint", "student.pt"]
if script.startswith("eval_"):
    sys.argv += ["--output", "unused"]
entrypoint.parse_args()
from legged_gym.envs import task_registry
assert "carrybox_teacher_warmup" in task_registry.task_classes
'''


class EntrypointImports(unittest.TestCase):
    def test_fresh_process_registration_before_argument_parsing(self):
        for script in ("eval_teacher_warmup_loco", "train_teacher_warmup_loco"):
            with self.subTest(script=script):
                old = subprocess.run([sys.executable, "-c", BOOTSTRAP,
                                      str(ROOT), script, "before_fix"],
                                     capture_output=True, text=True)
                self.assertNotEqual(old.returncode, 0)
                self.assertIn("partially initialized module", old.stderr)
                fixed = subprocess.run([sys.executable, "-c", BOOTSTRAP,
                                        str(ROOT), script, "fixed"],
                                       capture_output=True, text=True)
                self.assertEqual(fixed.returncode, 0, fixed.stderr)


if __name__ == "__main__":
    unittest.main()
