import types
import warnings

from logipy.logic.properties import LogipyPropertyException
import logipy.logic.properties as logipy_properties
from logipy.logic.timed_property_graph import *
from logipy.logic.monitored_predicate import *
from logipy.logic import prover
from logipy.monitor.time_source import global_stamp_and_increment


DISABLE_MONITORING_WHEN_PROPERTY_EXCEPTION_RAISED = True
property_exception_raised = False

_SPECIAL_NAMES = [
    '__abs__', '__add__', '__and__', '__call__', '__cmp__', '__coerce__',
    '__contains__', '__delitem__', '__delslice__', '__div__', '__divmod__',
    '__eq__', '__floordiv__', '__ge__', '__getitem__',
    '__getslice__', '__gt__', '__hex__', '__iadd__', '__iand__',
    '__idiv__', '__idivmod__', '__ifloordiv__', '__ilshift__', '__imod__',
    '__imul__', '__invert__', '__ior__', '__ipow__', '__irshift__',
    '__isub__', '__iter__', '__itruediv__', '__ixor__', '__le__',
    '__long__', '__lshift__', '__lt__', '__mod__', '__mul__', '__ne__',
    '__neg__', '__oct__', '__or__', '__pos__', '__pow__', '__len__',
    '__radd__', # comment to not create string errors
    '__float__', '__int__', '__bool__', '__hash__', '__str__',
    '__rand__', '__rdiv__', '__rdivmod__', '__reduce__', '__reduce_ex__',
    '__reversed__', '__rfloorfiv__', '__rlshift__', '__rmod__',
    '__rmul__', '__ror__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__',
    '__rtruediv__', '__rxor__', '__setitem__', '__setslice__', '__sub__',
    '__truediv__', '__xor__', '__next__', #'__repr__',
]

_PRIMITIVE_CONVERTERS = set(['__float__', '__int__', '__bool__', '__str__', '__hash__', '__len__'])

__logipy_past_warnings = set()


class LogipyMethod:
    """Wrapper for every callable object that should be monitored."""

    def __init__(self, method, parent_object=None):
        # Prohibit double wrapping of a callable.
        if isinstance(method, LogipyMethod):
            exception_text = "LogipyMethod cannot call an instance of itself. "
            exception_text += "Consider using logipy_call function instead."
            raise Exception(exception_text)

        self.__parent_object = parent_object
        self.__method = method  # Wrapped callable object.
        self.__doc__ = method.__doc__
        if hasattr(method, "__name__"):
            self.__name__ = method.__name__
        if hasattr(method, "__module__"):
            self.__module__ = method.__module__

    def __call__(self, *args, **kwargs):
        """Wrapper method for monitoring the calls on a callable object."""
        global property_exception_raised

        # A graph to include the state of caller, the state of args and the new steps.
        total_execution_graph = TimedPropertyGraph()

        # Monitor "call" predicate on parent object.
        if self.__parent_object is not None and isinstance(self.__parent_object, LogipyPrimitive):
            # logipy_properties.combine(properties, self.__parent_object.get_logipy_properties())
            # apply_method_rules(self.__method.__name__, self.__parent_object,
            #                    "call", args, kwargs)
            current_timestamp = Timestamp(global_stamp_and_increment())

            call_graph = Call(self.__method.__name__).convert_to_graph()
            call_graph.set_timestamp(current_timestamp)

            self.__parent_object.get_execution_graph().logical_and(call_graph, current_timestamp)
            if not property_exception_raised or \
                    not DISABLE_MONITORING_WHEN_PROPERTY_EXCEPTION_RAISED:
                prover.prove_set_of_properties(logipy_properties.get_global_properties(),
                                               self.__parent_object.get_execution_graph())

            total_execution_graph.logical_and(
                self.__parent_object.get_execution_graph(), current_timestamp)

        # Monitor "called by" predicate on arguments passed to current call.
        args_list = list(args)
        args_list.extend(kwargs.values())
        for arg in args_list:
            if isinstance(arg, LogipyPrimitive):
                # logipy_properties.combine(properties, arg.get_logipy_properties())
                # apply_method_rules(self.__method.__name__, arg, "called by", args, kwargs)

                # Add the called by predicate to the execution graphs of all arguments.
                current_timestamp = Timestamp(global_stamp_and_increment())

                called_by_graph = CalledBy(self.__method.__name__).convert_to_graph()
                called_by_graph.set_timestamp(current_timestamp)

                arg.get_execution_graph().logical_and(called_by_graph, current_timestamp)
                if not property_exception_raised or \
                        not DISABLE_MONITORING_WHEN_PROPERTY_EXCEPTION_RAISED:
                    prover.prove_set_of_properties(logipy_properties.get_global_properties(),
                                                   arg.get_execution_graph())

                total_execution_graph.logical_and(arg.get_execution_graph(), current_timestamp)

        # TODO: FIND THE BEST WAY TO DO THE FOLLOWING
        # Monitor "returned by" predicate.
        try:
            ret = self.__method(*args, **kwargs)
        except Exception as err:
            if not isinstance(err, prover.PropertyNotHoldsException):
                args = [arg.get_logipy_value() if isinstance(arg, LogipyPrimitive) else arg for arg
                        in args]
                kwargs = {key: arg.get_logipy_value() if isinstance(arg, LogipyPrimitive) else arg
                          for key, arg in kwargs.items()}
                ret = self.__method(*args, **kwargs)
                logipy_warning(
                    "A method " + self.__method.__name__ +
                    " was called a second time at least once by casting away LogipyPrimitive " +
                    "due to invoking the error: " + str(err))
            else:
                property_exception_raised = True
                raise err
        # apply_method_rules(self.__method.__name__, ret, "returned by", args, kwargs)
        ret = LogipyPrimitive(ret, total_execution_graph)
        current_timestamp = Timestamp(global_stamp_and_increment())

        returned_by_graph = ReturnedBy(self.__method.__name__).convert_to_graph()
        returned_by_graph.set_timestamp(current_timestamp)
        ret.get_execution_graph().logical_and(returned_by_graph, current_timestamp)
        if not property_exception_raised or \
                not DISABLE_MONITORING_WHEN_PROPERTY_EXCEPTION_RAISED:
            prover.prove_set_of_properties(logipy_properties.get_global_properties(),
                                           ret.get_execution_graph())

        return ret

    def __get__(self, instance, cls):
        return types.MethodType(self, instance) if instance else self


class LogipyPrimitive:

    __logipy_id_count = 0  # Counter for each instantiated LogipyPrimitive so far.

    def __init__(self, value, previous_execution_graph=None):
        if isinstance(value, LogipyPrimitive):
            # If given value is already a LogipyPrimitive, copy its execution graph.
            self.__logipy_value = value.__logipy_value
            self.execution_graph = value.get_execution_graph().get_copy()
            self.timestamp = value.timestamp
        else:
            # If given value is not a LogipyPrimitive, instantiate a new set of properties.
            self.__logipy_value = value
            self.execution_graph = TimedPropertyGraph()
            self.timestamp = global_stamp_and_increment()

        if previous_execution_graph is not None:
            previous_copy = previous_execution_graph.get_copy()
            previous_copy.logical_and(self.execution_graph,
                                      self.execution_graph.get_most_recent_timestamp())
            self.execution_graph = previous_copy

        self.__logipy_id = str(LogipyPrimitive.__logipy_id_count)
        LogipyPrimitive.__logipy_id_count += 1  # TODO: Make it thread safe.

    def get_logipy_id(self):
        return self.__logipy_id

    def get_logipy_value(self):
        return self.__logipy_value

    def get_timestamp(self):
        return self.timestamp

    def increase_time(self):
        self.timestamp += 1

    def get_execution_graph(self):
        return self.execution_graph

    def __getattr__(self, method_name):
        # TODO: Rewrite function with a single exit point.
        # Delegate attribute lookup from wrapper to the wrapped object.
        if hasattr(self.__logipy_value, method_name):
            if _is_callable(getattr(self.__logipy_value, method_name)):
                # Wrap callable attributes into a LogipyMethod wrapper.
                return LogipyMethod(getattr(self.__logipy_value, method_name), self)
            else:
                # Wrap non-callables into a LogipyPremitive wrapper.
                value = getattr(self.__logipy_value, method_name)
                if isinstance(value, LogipyPrimitive):
                    return value
                return LogipyPrimitive(value, self.__execution_graph)
        else:
            raise AttributeError()

        # def method(self, *args, **kw):
        #     return LogipyPrimitive(getattr(self.__logipy_value, method_name)(*args, **kw))
        # return method
        # return object.__getattribute__(self, method_name)

    # def __setattr__(self, key, value):
        # self.__dict__["_LogipyPrimitive__logipy_value"].__dict__[key] = value
        # self.__dict__[key] = value

    def __nonzero__(self):
        return bool(self.value())  # TODO: value() method?????

    # def __hash__(self):
    #     return hash(self.__logipy_value)

    def __repr__(self):
        return repr(self.__logipy_value) +" (" +", ".join(self.__execution_graph) + ")"


def logipy_call(method, *args, **kwargs):
    """Call a callable object inside a LogipyMethod wrapper.

    :param method: The callable object to be wrapped and called.
    :param args: Arguments to be passed to the callable object.
    :param kwargs: Keyword arguments to be passed to the callable object.

    # TODO: Bettern return comment.
    :return: Upon successful verification, returns the value returned
    by the callable.
    """
    if isinstance(method, LogipyMethod):
        return method(*args, **kwargs)
    return LogipyMethod(method)(*args, **kwargs)


def logipy_value(obj):
    if isinstance(obj, LogipyPrimitive):
        return obj.get_logipy_value()
    return obj


def error(message):
    raise Exception(message)


def logipy_warning(logipy_warning_message):
    if logipy_warning_message in __logipy_past_warnings:
        return
    __logipy_past_warnings.add(logipy_warning_message)
    warnings.warn(logipy_warning_message)


def _make_primitive_method(method_name):
    def method(self, *args, **kwargs):
        if method_name in _PRIMITIVE_CONVERTERS:
            return getattr(self.get_logipy_value(), method_name)(*args, **kwargs)
        return LogipyMethod(getattr(self.get_logipy_value(), method_name), self)(*args, **kwargs)

        # _apply_method_rules(method_name, self, call_rules, *args, **kwargs)
        # for arg in args:
        #     graph_logic = graph_logic.union(_properties(arg))
        #     _apply_method_rules(method_name, arg, call_rules, *args, **kwargs)
        # for arg in kwargs.values():
        #     graph_logic = graph_logic.union(_properties(arg))
        #     _apply_method_rules(method_name, arg, call_rules, *args, **kwargs)
        # ret = LogipyPrimitive(getattr(self.logipy_value(), method_name)(*args, **kwargs), graph_logic)
        # _apply_method_rules(method_name, ret, return_rules, *args, **kwargs)
        # return ret
    return method


def _is_callable(obj):
    """Returns true if given object is a callable.

    Provides support for wrapped objects.
    """
    if isinstance(obj, LogipyPrimitive):
        return callable(obj.get_logipy_value())
    return callable(obj)


for method_name in _SPECIAL_NAMES:
    setattr(LogipyPrimitive, method_name, _make_primitive_method(method_name))

# unbound_variable = LogipyPrimitive(None)
