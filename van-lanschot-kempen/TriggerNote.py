from ProtectedNote import SemiProtectedNote


class TriggerNote(SemiProtectedNote):
    coupon = True

    def __init__(
        self,
        maturity: int,
        nominal_value: float,
        protection_value: float,
        american: bool,
        coupon: float,
        coupon_barrier: float,
        trigger_barrier: float,
        memory_coupon: bool,
        settlement_barrier: float,
        coupon_days: list,
    ):
        super().__init__(maturity, nominal_value, protection_value, american)
        self.coupon = coupon
        self.coupon_barrier = coupon_barrier
        self.trigger_barrier = trigger_barrier
        self.memory_coupon = memory_coupon
        self.settlement_barrier = settlement_barrier

        self.missed_coupons = 0
        self.coupon_days = coupon_days

    @property
    def end_settlement(self) -> float:
        """
        Get the end settlement value of the note.

        Returns:
            float: The end settlement value.
        """
        # Get the value that determines whether the protection barrier
        # has been breached
        if self.american:
            comparing_value = self.lowest_value
        else:
            comparing_value = self.underlying_value

        # Return the nominal value if protected or the underlying value if not
        if comparing_value < self.protection_value:
            return self.underlying_value
        else:
            return self.nominal_value

    def coupon_moment(self) -> float:
        """
        Calculate the coupon moment of the note.

        Returns:
            float: The coupon moment value.
        """
        # If the underlying value is above the trigger,
        # there is early settlement
        if self.underlying_value > self.trigger_barrier:
            return self.get_early_settlement()

        # Else determine whether there is a coupon
        coupon_right = 0
        if self.underlying_value >= self.coupon_barrier:
            # If underlying is above the coupons barrier,
            # the coupon plus missed coupons is returned
            coupon_right += self.coupon + self.missed_coupons
            self.missed_coupons = 0
        else:
            if self.memory_coupon:
                # Add the coupon to the missed coupons
                self.missed_coupons += self.coupon

        return coupon_right

    def get_early_settlement(self) -> float:
        """
        Get the early settlement value of the note.

        Returns:
            float: The early settlement value.
        """
        return self.nominal_value
