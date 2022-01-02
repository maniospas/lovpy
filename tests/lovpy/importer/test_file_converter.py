import tempfile
from pathlib import Path
from unittest import TestCase

from lovpy.importer.file_converter import restore_path, BACKUP_FOLDER


class Test(TestCase):

    def test_restore_path_on_non_cwd_root(self):
        temp_dir: tempfile.TemporaryDirectory = tempfile.TemporaryDirectory()
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
