from ProtectedNote import FullProtectedNote
from collections import deque


class IndexGarantieNote(FullProtectedNote):
    maxmonths = 10
    dayspermonth = 21

    def __init__(
        self,
        maturity: float,
        nominal_value: float,
        protection_value: float,
        participation_level: float,
        maximum: float,
        middeling: bool,
    ):
        """
        Initialize an IndexGarantieNote object.

        Args:
            maturity (float): The maturity of the note.
            nominal_value (float): The nominal value of the note.
            protection_value (float): The protection value of the note.
            participation_level (float): The participation level of the note.
            maximum (float): The maximum value of the note.
            middeling (bool): Whether the note has middeling.
        """
        super().__init__(maturity, nominal_value, protection_value)
        self.participation_level = participation_level
        self.maximum = maximum
        self.middeling = middeling

        maxlen = self.maxmonths * self.dayspermonth
        self.previous_underlyings = deque(maxlen=maxlen)

    @property
    def underlying_value(self):
        """
        Get the underlying value of the note.

        Returns:
            float: The underlying value.
        """
        return self._underlying_value

    @underlying_value.setter
    def underlying_value(self, value):
        """
        Set the underlying value of the note.

        Args:
            value (float): The underlying value to set.
        """
        # Update value
        self._underlying_value = value

        # Store last 10 months of underlying values
        self.previous_underlyings.append(value)

    @property
    def end_settlement(self):
        """
        Get the end settlement value of the note.

        Returns:
            float: The end settlement value.
        """
        if self.middeling:
            comparing_value = sum(self.previous_underlyings) / len(
                self.previous_underlyings
            )
        else:
            comparing_value = self.underlying_value

        if comparing_value < self.nominal_value:
            return self.protection_value
        else:
            before_maximum = (
                self.protection_value
                / self.nominal_value
                * self.participation_level
                * self.underlying_value
            )

            if self.maximum is not None and before_maximum > self.maximum:
                return self.maximum
            else:
                return before_maximum
