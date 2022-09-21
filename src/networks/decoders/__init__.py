from .cadepth import *
from .ddvnet import *
from .diffnet import *
from .hrdepth import *
from .monodepth import *
from .superdepth import *

__all__ = (
        cadepth.__all__ +
        ddvnet.__all__ +
        diffnet.__all__ +
        hrdepth.__all__ +
        monodepth.__all__ +
        superdepth.__all__ +
        ['DECODERS']
)

DECODERS = {
    'monodepth': MonodepthDecoder,
    'hrdepth': HRDepthDecoder,
    'superdepth': SuperdepthDecoder,
    'cadepth': CADepthDecoder,
    'diffnet': DiffNetDecoder,
    'ddvnet': DDVNetDecoder,
}
