from abc import ABC
from typing import Generic, TypeVar

# TODO: Consider designing separate class for tabu list
TL = TypeVar('TL')  # Tabu list


class TabuSearch(ABC, Generic[TL]):
    _tabu_list: TL
