from abc import ABC, abstractmethod
from numbers import Number
from typing import Any, Optional, Dict, Set
import re

import pandas as pd

INF = float('inf')


class ItemFilter(ABC):
    @abstractmethod
    def filter(self, item: Any) -> bool:
        pass

    def __call__(self, item: Any) -> bool:
        return self.filter(item)


UserPreference = Dict[str, ItemFilter]


class RegexFilter(ItemFilter):
    """Filters based on whether regex pattern can be found in item"""

    def __init__(self, pattern: str):
        self.filter_re = re.compile(pattern)

    def filter(self, item: str) -> bool:
        return bool(self.filter_re.search(item))


class NumericalFilter(ItemFilter):
    """
    Filters based on numerical value. You can either set min and max values or
    set only a single value by keyword arguments.
    """

    def __init__(self,
                 min_value: Optional[Number] = None,
                 max_value: Optional[Number] = None):
        self.min_value = min_value if min_value is not None else -INF
        self.max_value = max_value if max_value is not None else INF
        assert self.min_value <= self.max_value, \
            f'Min value {self.min_value} is greater than max value {self.max_value}'

    def filter(self, item: Number) -> bool:
        return self.min_value <= item <= self.max_value


class SetFilter(ItemFilter):
    """Filters based on whether item is in set."""

    def __init__(self, values: Set[Any]):
        self.values = set(values)
        assert len(self.values) > 0, 'Set cannot be empty'

    def filter(self, item: Any) -> bool:
        return item in self.values


class NotFilter(ItemFilter):
    """
    Apply logical not to another ItemFilter. To i.e. exclude items in range
    [5, 10] from a numerical filter, you can do:
    >>> not_filter = NotFilter(NumericalFilter(5, 10))
    >>> df[some_numerical_column].apply(not_filter)
    """

    def __init__(self, item_filter: ItemFilter):
        self.item_filter = item_filter
        assert isinstance(item_filter, ItemFilter), \
            'Supplied argument is not a ItemFilter'

    def filter(self, item: Any) -> bool:
        return not self.item_filter(item)


def filter_on_user_pref(user_pref: Optional[UserPreference],
                        df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Filter a dataframe based on user preferences. If user_pref is None,
    return the original dataframe.
    """
    if user_pref is None:
        return df_train

    df_user = df_train.copy()
    for col, item_filter in user_pref.items():
        df_user = df_user[df_user[col].apply(item_filter)]
    return df_user
