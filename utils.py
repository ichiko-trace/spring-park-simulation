"""
Shared utility functions for LLM-based agent in 2D worlds with multiple places.
"""
from typing import Tuple, Optional, List, Dict, TypedDict


class FireConfig(TypedDict, total=False):
    """Type definition for fire event configuration"""
    name: str  # Fire name (required)
    start_step: int  # Step at which fire appears (required)
    intensity: float  # Fire intensity (0.0 to 1.0) (required)
    radius: int  # Perception radius (required)
    center_x: int  # Fire position X (optional, random if omitted)
    center_y: int  # Fire position Y (optional, random if omitted)


class PlaceConfig(TypedDict):
    """Type definition for place configuration"""
    name: str  # Place name (required)
    type: str  # Place type: bar, cafe, library, etc. (required)
    center_x: int  # X coordinate of place center (required)
    center_y: int  # Y coordinate of place center (required)
    half_size: int  # Half size of the place (required)
    capacity: int  # Maximum comfortable capacity of the place (required)


def is_position_in_place(
    position: Tuple[int, int],
    half_size: int,
    center_x: int = 0,
    center_y: int = 0
) -> bool:
    """
    Check if a position is inside a place area.
    Place is centered at (center_x, center_y).

    Args:
        position: (x, y) coordinates to check
        half_size: Half size of the place (place covers -half_size to +half_size from center)
        center_x: X coordinate of place center (default: 0)
        center_y: Y coordinate of place center (default: 0)

    Returns:
        True if position is inside the place, False otherwise
    """
    x, y = position
    # Place covers -half_size to +half_size (inclusive on both ends) from center
    return (center_x - half_size <= x <= center_x + half_size and
            center_y - half_size <= y <= center_y + half_size)


def get_place_at_position(
    position: Tuple[int, int],
    places: List[PlaceConfig]
) -> Optional[PlaceConfig]:
    """
    Get the place that contains the given position.

    Args:
        position: (x, y) coordinates to check
        places: List of place configurations, each with 'center_x', 'center_y', 'half_size'

    Returns:
        Place dictionary if position is in a place, None otherwise
    """
    for place in places:
        if is_position_in_place(
            position,
            place['half_size'],
            place['center_x'],
            place['center_y']
        ):
            return place
    return None
