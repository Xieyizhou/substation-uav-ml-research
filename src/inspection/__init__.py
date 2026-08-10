"""Read-only project inspection services for the local research sandbox."""

from src.inspection.config import InspectionConfig

__all__ = ("InspectionConfig", "InspectionService")


def __getattr__(name):
    if name == "InspectionService":
        from src.inspection.service import InspectionService

        return InspectionService
    raise AttributeError(name)
