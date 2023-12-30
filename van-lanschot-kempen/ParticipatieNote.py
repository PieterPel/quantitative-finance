from ProtectedNote import SemiProtectedNote


class ParticipatieNote(SemiProtectedNote):
    def __init__(
        self,
        maturity: float,
        nominal_value: float,
        protection_value: float,
        american: bool,
        participation_level: float,
    ):
        """
        Initialize a ParticipatieNote object.

        Args:
            participation_level: The participation level of the note.
            american: Whether the note is American style.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(maturity, nominal_value, protection_value, american)
        self.participation_level = participation_level

    @property
    def end_settlement(self):
        """
        Get the end settlement value of the note.

        Returns:
            float: The end settlement value.
        """
        # If the underlying value is above the nominal value
        if self.underlying_value > self.nominal_value:
            return (
                self.nominal_value
                + self.participation_level
                * (self.underlying_value - self.nominal_value)
                / self.nominal_value
            )

        # If the underlying value is below the nominal value

        # Get the value that determines whether the protection
        # barrier has been breached
        if self.american:
            comparing_value = self.lowest_value
        else:
            comparing_value = self.underlying_value

        # Return the nominal value if protected or the underlying value if not
        if comparing_value < self.protection_value:
            return self.underlying_value
        else:
            return self.nominal_value
