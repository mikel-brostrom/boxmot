# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from pathlib import Path
from boxmot.utils import logger as LOGGER

def get_postprocessor(name):
    """
    Factory function to get the post-processing function by name.
    Lazy loads the module to avoid unnecessary imports.
    """
    if name == "gsi":
        from boxmot.postprocessing.gsi import gsi
        return gsi
    elif name == "gbrc":
        from boxmot.postprocessing.gbrc import gbrc
        return gbrc
    elif name == "sct":
        from boxmot.postprocessing.sct import sct
        return sct
    else:
        return None

def apply_postprocessing(opt, exp_dir):
    """
    Applies the selected post-processing method.
    """
    method = getattr(opt, "postprocessing", "none")
    if method == "none":
        return

    postprocessor = get_postprocessor(method)
    if postprocessor is None:
        LOGGER.warning(f"Unknown postprocessing method: {method}")
        return

    LOGGER.opt(colors=True).info(f"<cyan>[3b/4]</cyan> Applying {method.upper()} postprocessing...")

    if method == "sct":
        dets_dir = opt.project / 'dets_n_embs' / opt.yolo_model[0].stem / 'dets'
        embs_dir = opt.project / 'dets_n_embs' / opt.yolo_model[0].stem / 'embs' / opt.reid_model[0].stem
        postprocessor(exp_dir, dets_dir, embs_dir)
    else:
        postprocessor(mot_results_folder=exp_dir)
