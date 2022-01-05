import sys
import re
from traceback import StackSummary, FrameSummary, TracebackException, extract_tb

from .exceptions import PropertyNotHoldsException
from .lovpy_utils import get_lovpy_system_files


def lovpy_exception_handler(ex_type, value, tb) -> None:
    file = sys.stderr

    exception_stack_summary: StackSummary = _clean_summary_from_lovpy_files(extract_tb(tb))

    if ex_type is PropertyNotHoldsException and value.last_proved_stacktrace:
        last_proved_stacktrace = _clean_stacktrace_from_lovpy_files(value.last_proved_stacktrace)
        exception_stack_summary = _add_last_proved_info_to_stack_summary(exception_stack_summary,
                                                                         last_proved_stacktrace)

    exception_stack_summary = _clean_summary_from_monitor_calls(exception_stack_summary)
    exception_stack_summary = _clean_summary_from_file_modifications(exception_stack_summary)
    exception_stack_summary = _clean_summary_from_initial_runpy(exception_stack_summary)

    # Rebuild TracebackException to contain the updated stacktrace.
    tb_ex = TracebackException(ex_type, value, tb)
    tb_ex.stack = exception_stack_summary

    for line in tb_ex.format():
        print(line, file=file, end="")


def lovpy_dev_exception_handler(ex_type, value, tb):
    """Exception handler that doesn't hide lovpy's internal errors, used for dev purposes."""
    file = sys.stderr

    # print(dir(tb))
    # print(dir(tb.tb_frame))

    exception_stack_summary = extract_tb(tb)

    if ex_type is PropertyNotHoldsException and value.last_proved_stacktrace:
        exception_stack_summary = _add_last_proved_info_to_stack_summary(
            exception_stack_summary, value.last_proved_stacktrace)

    exception_stack_summary = _clean_summary_from_monitor_calls(exception_stack_summary)
    exception_stack_summary = _clean_summary_from_file_modifications(exception_stack_summary)

    # Rebuild TracebackException to contain the updated stacktrace.
    tb_ex = TracebackException(ex_type, value, tb)
    tb_ex.stack = exception_stack_summary

    for line in tb_ex.format():
        print(line, file=file, end="")


def _clean_summary_from_lovpy_files(summary: StackSummary):
    lovpy_files = {str(p) for p in get_lovpy_system_files()}

    clean_summary = StackSummary()

    for frame in summary:
        if frame.filename not in lovpy_files:
            clean_summary.append(frame)

    return clean_summary


def _clean_summary_from_monitor_calls(summary: StackSummary):
    clean_summary = StackSummary()
    
    for frame in summary:
        line = frame.line

        # Remove all wrapper calls to lovpy_call().
        if line:
            matches = re.match("^(.*)lovpy_call[(]globals[(][)], locals[(][)], (.*)[)](.*)", line)
            if matches:
                call_parts = matches.groups()[1].split(sep=", ")
                function_call = call_parts.pop(0)
                if call_parts:
                    function_call = "{}({})".format(function_call, ", ".join(call_parts))
                else:
                    function_call = "{}()".format(function_call)
                line = "".join([matches.groups()[0], function_call, matches.groups()[2]])

        clean_summary.append(FrameSummary(frame.filename, frame.lineno,
                                          frame.name, line=line))
    
    return clean_summary


def _clean_summary_from_file_modifications(summary: StackSummary):
    clean_summary = StackSummary()
    clean_summary.append(summary[0])  # The entry point is not currently modified by lovpy.

    for frame in summary[1:]:
        original_lineno = frame.lineno - 1  # 1 line added to the top for monitoring

        # Fixed proved indicator line.
        name = frame.name
        proved_label = "<-- LAST CORRECT LINE, line "
        pos = name.find(proved_label)
        if pos > -1:
            proved_lineno = int(name[pos+len(proved_label):]) - 1
            name = name[:pos+len(proved_label)] + str(proved_lineno)

        clean_summary.append(FrameSummary(frame.filename, original_lineno,
                                          name, line=frame.line))

    return clean_summary


def _clean_summary_from_initial_runpy(summary: StackSummary):
    clean_summary = StackSummary()

    found_non_runpy = False

    for frame in summary:
        if frame.filename.endswith("runpy.py") and not found_non_runpy:
            continue
        else:
            found_non_runpy = True
            clean_summary.append(frame)

    return clean_summary


def _clean_stacktrace_from_lovpy_files(stacktrace):
    lovpy_files = {str(p) for p in get_lovpy_system_files()}
    return [f for f in stacktrace if f.filename not in lovpy_files]


def _add_last_proved_info_to_stack_summary(stack_summary: StackSummary, proved_stacktrace):
    for proved_frame in proved_stacktrace:
        for i, frame in enumerate(stack_summary):
            if proved_frame.filename == frame.filename and proved_frame.function == frame.name:
                verbose_name = "{} <-- LAST CORRECT LINE, line {}".format(frame.name,
                                                                          proved_frame.lineno)
                stack_summary[i] = FrameSummary(frame.filename, frame.lineno,
                                                verbose_name, line=frame.line)
                return stack_summary
