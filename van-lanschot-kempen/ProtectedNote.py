class ProtectedNote:
    """
    A class representing a protected note.
    """

    coupon = False

    def __init__(self, maturity, nominal_value, protection_value):
        """
        Initialize a ProtectedNote object.

        Args:
            maturity (int): The maturity of the note.
            nominal_value (float): The nominal value of the note.
            protection_value (float): The protection value of the note.
        """
        self.maturity = maturity
        self.nominal_value = self.underlying_value = nominal_value
        self.protection_value = protection_value

    @property
    def end_settlement(self):
        """
        Get the end settlement value of the note.
        """
        pass


class SemiProtectedNote(ProtectedNote):
    """
    A class representing a semi-protected note.
    """

    def __init__(self, maturity, nominal_value, protection_value, american):
        """
        Initialize a SemiProtectedNote object.

        Args:
            maturity (int): The maturity of the note.
            nominal_value (float): The nominal value of the note.
            protection_value (float): The protection value of the note.
            american (bool): Whether the note is American-style or not.
        """
        self.lowest_value = nominal_value
        super().__init__(maturity, nominal_value, protection_value)
        self.american = american

    @property
    def underlying_value(self):
        """
        Get the underlying value of the note.
        """
        return self._underlying_value

    @underlying_value.setter
    def underlying_value(self, value):
        """
        Set the underlying value of the note.

        Args:
            value (float): The new underlying value.
        """
        # Set the underlying value
        self._underlying_value = value

        # Update the lowest value
        self._update_lowest_value()

    def _update_lowest_value(self):
        """
        Update the lowest value of the note.
        """
        self.lowest_value = min(self.lowest_value, self.underlying_value)


class FullProtectedNote(ProtectedNote):
    """
    A class representing a fully protected note.
    """
