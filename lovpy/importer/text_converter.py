SPECIALS = '+/%,!?:;"()<>[]#$=-/\n '  # All special characters of python.

# All keywords of python.
KEYWORDS = ['range', 'print', 'False', 'None', 'True', 'and', 'as', 'assert', 'async',
            'await', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else',
            'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
            'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while',
            'with', 'yield']

LOVPY_KEYWORDS = []  # All newly introduced keywords by lovpy.


def transform_lines(lines):
    """Transforms lines of python code into a lovpy testable unit."""
    new_lines = ["from lovpy import LogipyPrimitive, lovpy_call\n"]

    for line in lines:
        if (line.startswith("import ") or line.startswith("from ") or
                line.strip().startswith("def ") or line.strip().startswith("class ")):
            # Nothing to change on imports and functions/classes definition.
            new_lines.append(line)
        else:
            new_line = ""
            primitive = ""

            for c in line:
                # Parse each line, primitive by primitive.
                if c in SPECIALS:  # When a special character is found, primitive is over.
                    if (c == '('
                            and primitive
                            and primitive not in LOVPY_KEYWORDS
                            and "." != primitive[0]):
                        # Pass all callable objects as the first argument to a lovpy_call call.
                        primitive = "lovpy_call(globals(), locals(), " + primitive + ", "
                        c = ""
                    # elif primitive and primitive not in keywords:
                    #     primitive = "LogipyPrimitive("+primitive+")"
                    new_line += primitive + c
                    primitive = ""
                else:  # While not encountering a special character, keep building the primitive.
                    primitive += c

            if primitive:
                # TODO: Is this branch ever reachable? '\n' should normally terminate each line.
                new_line += "LogipyPrimitive(" + primitive + ")"

            new_lines.append(new_line)

    return new_lines
