import re

from logipy.logic.timed_property_graph import *


class MonitoredPredicate:
    def __init__(self, *args):
        self.args = list(args)

    def convert_to_graph(self):
        return PredicateGraph(
            self.get_predicate_name(), *self.args, MonitoredVariable("VAR"))

    def add_args(self, args):
        self.args.extend(args)

    def match_in_text(self, text):
        return self.get_regex().match(text)

    def get_regex(self):
        """Subclass and implement."""
        raise Exception("Subclass and implement.")

    def get_predicate_name(self):
        """Subclass and implement."""
        raise Exception("Subclass and implement.")

    @staticmethod
    def find_text_matching_monitored_predicate(text):
        """
        Searches for any monitored predicate matching the given text.

        :param text: The text to be matched.
        :return: If any matching monitored predicate is found, an instance of that
                monitored predicate is returned. If no matching monitored predicate is found,
                None is returned.
        """
        matching_function = None
        for function in MonitoredPredicate.__subclasses__():
            function_obj = function()
            matching_obj = function_obj.match_in_text(text)
            if matching_obj:
                matching_function = function_obj
                matching_function.add_args(matching_obj.groups())
        return matching_function


class Call(MonitoredPredicate):
    def get_predicate_name(self):
        return "call"

    def get_regex(self):
        return re.compile(r"^call (?P<call_arg>\S*)$")


class ReturnedBy(MonitoredPredicate):
    def get_predicate_name(self):
        return "returned by"

    def get_regex(self):
        return re.compile(r"^returned by (?P<returned_by_arg>\S*)$")


class CalledBy(MonitoredPredicate):
    def get_predicate_name(self):
        return "called by"

    def get_regex(self):
        return re.compile(r"^called by (?P<called_by_arg>\S*)$")
