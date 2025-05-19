from enum import Enum

class SideType(Enum):
    CT = "ct"
    T = "t"

    @classmethod
    def from_str(cls, side_str: str) -> 'SideType':
        """Creates an instance of the enum from a string, raising a ValueError if the string is not a valid types."""
        if side_str.lower() == "ct":
            return cls.CT
        elif side_str.lower() == "t":
            return cls.T
        else:
            raise ValueError(f"Invalid side type: {side_str}")
    