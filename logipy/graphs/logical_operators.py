class LogicalOperator:
    def __init__(self, *args):
        self.args_list = [arg.__repr__() for arg in args]
        if not self.operands_order_matters():
            # Sort the arguments, so their order doesn't matter in string representation.
            self.args_list.sort()
        # String representation based on the textual representation of operands,
        # is meant to represent two different operators that act on the same operands,
        # in the same way. Later, may consider to use another structure or another
        # hashing function.
        if len(self.args_list) == 1:
            self.str_representation = self.get_operator_symbol() + "(" + self.args_list[0] + ")"
        else:
            self.str_representation = (self.get_operator_symbol()
                                       + "(" + ",".join(self.args_list) + ")")

    # def __str__(self):
    #     return self.str_representation

    # def __hash__(self):
    #     if self.hashing_by_name_is_disabled:
    #         return
    #     else:
    #         return hash(self.str_representation)

    def __repr__(self):
        # Addition of hash is performed because graphviz utilizes str() to identify nodes.
        # So, nodes whose str() returns the same value are treated as equal.
        return self.str_representation + "-->" + str(hash(self))

    def __copy__(self):
        return type(self)(*self.args_list)

    def operands_order_matters(self):
        return False

    def get_operator_symbol(self):
        raise Exception("Subclass and implement.")

    def logically_matches(self, other):
        return type(self) is type(other)


class AndOperator(LogicalOperator):
    def get_operator_symbol(self):
        return "AND"


class NotOperator(LogicalOperator):
    def get_operator_symbol(self):
        return "NOT"


class ImplicationOperator(LogicalOperator):
    def get_operator_symbol(self):
        return "-->"

    def operands_order_matters(self):
        return True
