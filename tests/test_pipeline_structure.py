"""Dependency-free checks for the package boundary and launcher."""
import ast
import contextlib
import importlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from skysim_gw import cli

ROOT = Path(__file__).resolve().parents[1]

class PipelineStructureTests(unittest.TestCase):
    def test_list_without_scientific_imports(self):
        with patch.object(cli.importlib, 'import_module') as run, contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main(['--list']), 0)
        run.assert_not_called()
        self.assertIn('doublet-parallel', output.getvalue())

    def test_dry_run_preserves_argument_boundaries(self):
        with patch.object(cli.importlib, 'import_module') as run, contextlib.redirect_stdout(io.StringIO()) as output:
            cli.main(['--dry-run', 'detect', '/a path/input.npz', '/output.txt'])
        run.assert_not_called()
        self.assertIn("'/a path/input.npz'", output.getvalue())

    def test_dispatch_and_restore_argv(self):
        old = sys.argv
        observed = []
        from types import SimpleNamespace
        def load(module):
            return SimpleNamespace(main=lambda: observed.append((module, sys.argv.copy())))
        with patch.object(cli.importlib, 'import_module', side_effect=load):
            cli.main(['detect', 'input.npz', 'output.txt'])
        self.assertIs(sys.argv, old)
        self.assertEqual(observed[0], ('skysim_gw.stages.detect', ['skysim_gw.stages.detect', 'input.npz', 'output.txt']))

    def test_reject_options_that_would_be_silently_ignored(self):
        with patch.object(cli.importlib, 'import_module') as run, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli.main(['check-snr', '--wrong-option'])
        run.assert_not_called()

    def test_legacy_import_is_same_module(self):
        self.assertIs(importlib.import_module('legacy.gwdali_lvk_network'), importlib.import_module('skysim_gw.detectors'))

    def test_all_stage_modules_exist(self):
        for module, _ in cli.STAGES.values():
            self.assertTrue((ROOT / (module.replace('.', '/') + '.py')).is_file())

    def test_package_has_no_legacy_or_diagnostic_imports_in_production(self):
        legacy = {p.stem for p in (ROOT / 'legacy').glob('*.py')}
        for path in (ROOT / 'skysim_gw').rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn(node.module, legacy, str(path))
                    if 'diagnostics' not in path.parts:
                        self.assertFalse((node.module or '').startswith('skysim_gw.diagnostics'), str(path))

    def test_preserved_defaults_and_shared_paths(self):
        from skysim_gw.settings import common, snr, localize, population, catalogue
        self.assertEqual(snr.BASE_DIR, common.BASE_DIR)
        self.assertEqual(population.SOURCE_FILE.parent, common.BASE_DIR)
        self.assertEqual(localize.NPOINTS, 300)
        self.assertEqual(localize.MAX_NEW_SOURCES, 9)
        self.assertEqual(catalogue.ZF, 3.5)  # Deliberately preserved standalone example.
        self.assertEqual(snr.NETWORK_SNR_THRESHOLD, 12.0)

if __name__ == '__main__':
    unittest.main()
