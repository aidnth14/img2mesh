from dataclasses import dataclass, field
from pathlib import Path

_ALLOWED_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class Img2MeshError(Exception):
    """Base exception for img2mesh errors."""


class ConfigurationError(Img2MeshError):
    """Invalid configuration value."""


class InputError(Img2MeshError):
    """Invalid or unreadable input image."""


class OutputError(Img2MeshError):
    """Invalid output path or format."""


class ReconstructionError(Img2MeshError):
    """3D reconstruction failure."""


class ExportError(Img2MeshError):
    """Mesh export failure."""


@dataclass(frozen=True, slots=True)
class AppConfig:
    log_level: str = "INFO"
    max_image_pixels: int = 100_000_000
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "img2mesh")
    keep_intermediate_files: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.log_level, str):
            raise ConfigurationError("Log level must be a string.")

        normalized_level = self.log_level.upper()
        if normalized_level not in _ALLOWED_LOG_LEVELS:
            allowed = ", ".join(sorted(_ALLOWED_LOG_LEVELS))
            raise ConfigurationError(
                f"Invalid log level '{self.log_level}'. Allowed levels: {allowed}."
            )
        object.__setattr__(self, "log_level", normalized_level)

        if (
            isinstance(self.max_image_pixels, bool)
            or not isinstance(self.max_image_pixels, int)
            or self.max_image_pixels <= 0
        ):
            raise ConfigurationError("Maximum image pixels must be a positive integer.")

        try:
            normalized_cache_dir = Path(self.cache_dir).expanduser()
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                "Cache directory must be a valid filesystem path."
            ) from exc
        object.__setattr__(self, "cache_dir", normalized_cache_dir)

        if not isinstance(self.keep_intermediate_files, bool):
            raise ConfigurationError("keep_intermediate_files must be True or False.")


DEFAULT_CONFIG = AppConfig()
