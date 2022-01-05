import runpy
import sys
from pathlib import Path

from lovpy.logic.properties import RuleSet, add_global_rule_set
from lovpy.logic.next_theorem_selectors import NextTheoremSelector
from lovpy.importer.file_converter import convert_path, restore_path


class InvalidConversionPath(RuntimeError):
    pass


class Program:
    def __init__(self, entry_point: Path, config: 'VerificationConfiguration'):
        """Creates a new verifiable program.

        :param entry_point: Path to the entry point script.
        :type entry_point: Path
        :param config: Configuration of verification parameters.
        :type config: VerificationConfiguration
        """
        self.entry_point: Path = entry_point
        self.config: 'VerificationConfiguration' = config
        self.rule_sets: set[RuleSet] = set()

    def __call__(self, *argv) -> None:
        """Runs the program under verification.

        Entry point of the program should be located under the directory tree
        to be converted by the preprocessor. Root of the directory tree to be
        converted is specified via self.config attribute.

        Default root directory of conversion tree is the current working dir.

        If relative paths are provided, they are converted to absolute ones
        based on current working directory.

        :raises InvalidConversionPath: When conversion path does not include
                entry point of the program.
        """
        if not self.entry_point.absolute().is_relative_to(self.config.conversion_root.absolute()):
            message = "Lovpy requires entry point to be located under conversion root.\n"
            message += f"Entry point: {str(self.entry_point)}\n"
            message += f"Conversion root: {str(self.config.conversion_root)}"
            raise InvalidConversionPath(message)

        convert_path(self.config.conversion_root)
        old_argv = sys.argv
        sys.argv = [str(self.entry_point)]
        sys.argv.extend(list(argv))

        for group in self.rule_sets:
            add_global_rule_set(group)

        try:
            runpy.run_path(str(self.entry_point), run_name="__main__")
        finally:
            sys.argv = old_argv
            restore_path(self.config.conversion_root)

    def add_monitored_rules(self, rules: RuleSet) -> None:
        """Adds a set of rules to be monitored.

        :param rules: Rules group to be monitored.
        :type rules: RuleSet
        """
        self.rule_sets.add(rules)


class VerificationConfiguration:
    def __init__(self, provers: list[NextTheoremSelector]):
        self.provers: list[NextTheoremSelector] = provers
        self.conversion_root: Path = Path().cwd()
