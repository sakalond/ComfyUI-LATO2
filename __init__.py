"""ComfyUI-LATO2: LATO.2 low-poly re-meshing as a ComfyUI node."""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"  # the "upload mesh" button (web/lato2_upload.js)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
