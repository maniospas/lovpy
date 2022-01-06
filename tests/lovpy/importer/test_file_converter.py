from shutil import copytree
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest import TestCase

from lovpy.importer.file_converter import convert_path, restore_path, BACKUP_FOLDER


class Test(TestCase):

    def test_convert_path_with_non_ascii_file(self):
        temp_dir, test_dir = Test.copytree_to_tempdir(Path(__file__).parent /
                                                      Path("test_utf8_support"))
        convert_path(test_dir)

        for file in test_dir.glob("*.py"):
            text = file.read_text(encoding="utf-8")
            self.assertIn("lovpy", text)

    def test_restore_path_on_non_cwd_root(self):
        temp_dir: TemporaryDirectory = TemporaryDirectory()
        backup_dir: Path = Path(temp_dir.name) / BACKUP_FOLDER
        backup_dir.mkdir()
        file: Path = backup_dir / "file1.py"
        file.write_text("test")

        restored_file = Path(temp_dir.name) / "file1.py"
        self.assertTrue(file.exists())
        self.assertFalse(restored_file.exists())

        restore_path(Path(temp_dir.name))

        self.assertFalse(file.exists())
        self.assertTrue(restored_file.exists())

    @staticmethod
    def copytree_to_tempdir(src: Path) -> tuple[TemporaryDirectory, Path]:
        temp_dir: TemporaryDirectory = TemporaryDirectory()
        copied_folder: Path = Path(temp_dir.name) / src.name
        copytree(src, copied_folder)
        return temp_dir, copied_folder
