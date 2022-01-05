import hashlib
from unittest import TestCase
from pathlib import Path
from tempfile import TemporaryDirectory
from shutil import copytree
import sys
import os

from lovpy.logic.properties import RuleSet
from lovpy.monitor.program import Program, VerificationConfiguration, InvalidConversionPath
from lovpy.importer.gherkin_importer import GherkinImporter
from lovpy.exceptions import PropertyNotHoldsException
from lovpy.logic.next_theorem_selectors import BetterNextTheoremSelector


class TestProgram(TestCase):

    def setUp(self) -> None:
        self.temp_dir: TemporaryDirectory = TemporaryDirectory()
        test_programs_folder: Path = Path(__file__).parent / Path("test_programs")
        self.programs_path: Path = Path(self.temp_dir.name) / test_programs_folder.name
        copytree(test_programs_folder, self.programs_path)

    def test_init(self) -> Program:
        entry_point: Path = self.programs_path / "invalid_counter_test.py"
        config: VerificationConfiguration = VerificationConfiguration(
            provers=[BetterNextTheoremSelector()]
        )
        program: Program = Program(entry_point, config)
        self.assertEqual(program.entry_point, entry_point)
        self.assertEqual(program.config, config)

        return program

    def test_add_monitored_rules(self) -> Program:
        program: Program = self.test_init()

        rules: list[RuleSet] = GherkinImporter().add_import_path(
            self.programs_path / "invalid_counter_rules.gherkin").import_rules()
        for r in rules:
            program.add_monitored_rules(r)

        self.assertEqual(len(program.rule_sets), 1)

        return program

    def test_call(self) -> None:
        program: Program = self.test_add_monitored_rules()

        conversion_root: Path = self.programs_path
        program.config.conversion_root = conversion_root

        old_argv: list = sys.argv
        original_hashes: dict[Path, str] = dict()
        for file in conversion_root.rglob("*.*"):
            original_hashes[file] = TestProgram.calculate_file_blake2b(file)

        self.assertRaises(PropertyNotHoldsException, program.__call__)

        for file, original_hash in original_hashes.items():
            new_hash: str = TestProgram.calculate_file_blake2b(file)
            self.assertEqual(original_hash, new_hash)

        self.assertEqual(old_argv, sys.argv)

    def test_call_with_invalid_conversion_root(self) -> None:
        entry_point: Path = Path(__file__).parent / Path("test_programs/invalid_counter_test.py")
        config: VerificationConfiguration = VerificationConfiguration(
            provers=[BetterNextTheoremSelector()]
        )
        config.conversion_root = Path(__file__).parent / Path("test_programs/empty_folder")

        program: Program = Program(entry_point, config)
        self.assertRaises(InvalidConversionPath, program.__call__)

    def test_call_with_relative_script_absolute_conversion_root(self) -> None:
        program: Program = self.test_add_monitored_rules()
        program.entry_point = program.entry_point.relative_to(self.programs_path)

        conversion_root: Path = self.programs_path
        conversion_root = conversion_root.absolute()
        program.config.conversion_root = conversion_root

        old_argv: list = sys.argv
        original_hashes: dict[Path, str] = dict()
        for file in conversion_root.rglob("*.*"):
            original_hashes[file] = TestProgram.calculate_file_blake2b(file)

        old_cwd = os.getcwd()
        os.chdir(self.programs_path)

        self.assertRaises(PropertyNotHoldsException, program.__call__)

        os.chdir(old_cwd)

        for file, original_hash in original_hashes.items():
            new_hash: str = TestProgram.calculate_file_blake2b(file)
            self.assertEqual(original_hash, new_hash)

        self.assertEqual(old_argv, sys.argv)

    @staticmethod
    def calculate_file_blake2b(path: Path) -> str:
        hasher = hashlib.blake2b()
        with path.open('rb') as file:
            chunk = file.read(8192)
            while chunk:
                hasher.update(chunk)
                chunk = file.read(8192)
        return hasher.hexdigest()


class TestVerificationConfiguration(TestCase):

    def test_init(self):
        config: VerificationConfiguration = VerificationConfiguration(
            [BetterNextTheoremSelector()]
        )

        self.assertEqual(len(config.provers), 1)
        self.assertIsInstance(config.provers[0], BetterNextTheoremSelector)
        self.assertEqual(config.conversion_root, Path.cwd())
