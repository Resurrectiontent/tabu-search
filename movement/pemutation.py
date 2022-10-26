from abc import ABC

from movement.base import MovementBehaviour


class PermutationMovement(MovementBehaviour, ABC):
    @staticmethod
    def _permute():
    ...
