class SolutionId:
    def __init__(self, parent_name: str, *solution_idx: str):
        self._str = f'{parent_name}({",".join(solution_idx)})' if len(solution_idx) else parent_name
        self._hash = hash(self._str)

    def __eq__(self, other: 'SolutionId'):
        assert issubclass(type(other), SolutionId), 'Can only check equality of SolutionId with another SolutionId' \
                                                    f' ({type(other).__name__} was passed)'
        return hash(self) == hash(other)

    def __hash__(self):
        return self._hash

    def __str__(self):
        return self._str
