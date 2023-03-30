from enum import Enum


class VRepRapStates(Enum):
    """
    States the printer can be in when making a VRepRapMove
    """
    PRINTING = 1  # Prints/Extrudes a line
    TRAVELING = 2
    EXTRUDING = 3  # Just squeezes out some material without moving the print head
    RETRACTING = 4  # Opposite of EXTRUDING
    CHANGING_PARAM = 5  # Printer changes some non monitored parameter
    NOZZLE_LIFT = 6  # nozzle is lifted or lowered in z direction only (prior to LIFTED_TRAVELING)
    LIFTED_TRAVELING = 7  # nozzle lifted and is now in a horizontal travel
    NOZZLE_LOWER = 8
    STANDBY = 9
