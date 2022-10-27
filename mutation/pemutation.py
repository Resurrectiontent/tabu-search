from abc import ABC

from mutation.base import MutationBehaviour


class PermutationMutation(MutationBehaviour, ABC):
    @staticmethod
    def _permute():
        ...
