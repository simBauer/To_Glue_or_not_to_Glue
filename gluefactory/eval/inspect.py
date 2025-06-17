# Code from gluefactory (https://github.com/cvg/glue-factory)
# -----------------------------------------------------------
# Modified by Simone Gaisbauer: 
# - Modified to load pairs file from config for TUM dataset (car or UAV pairs)
# -----------------------------------------------------------

import argparse
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from omegaconf import OmegaConf

import matplotlib
import matplotlib.pyplot as plt

from ..settings import EVAL_PATH
from ..visualization.global_frame import GlobalFrame
from ..visualization.two_view_frame import TwoViewFrame
from . import get_benchmark
from .eval_pipeline import load_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", type=str)
    parser.add_argument("--x", type=str, default=None)
    parser.add_argument("--y", type=str, default=None)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument(
        "--default_plot", type=str, default=TwoViewFrame.default_conf["default"]
    )

    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()

    output_dir = Path(EVAL_PATH, args.benchmark)

    results = {}
    summaries = defaultdict(dict)

    predictions = {}

    if args.backend:
        matplotlib.use(args.backend)

    bm = get_benchmark(args.benchmark)
    if (bm.__name__).endswith("TumFacadePipeline"):
        # Allow different pairs files to be loaded from config
        pairs = []
        for name in args.dotlist:
            saved_conf = OmegaConf.load(output_dir / name / "conf.yaml")
            pairs.append(saved_conf.data.pairs)
        # Check if all elements are the same
        if all(x == pairs[0] for x in pairs):
            pairs = pairs[0]
        else:
            raise ValueError("Pairs of the configs are not the same")
        loader = bm.get_dataloader(pairs=pairs)
    else:
        loader = bm.get_dataloader()

    for name in args.dotlist:
        experiment_dir = output_dir / name
        pred_file = experiment_dir / "predictions.h5"
        s, results[name] = load_eval(experiment_dir)
        predictions[name] = pred_file
        for k, v in s.items():
            summaries[k][name] = v

    pprint(summaries)

    plt.close("all")

    frame = GlobalFrame(
        {"child": {"default": args.default_plot}, **vars(args)},
        results,
        loader,
        predictions,
        child_frame=TwoViewFrame,
    )
    frame.draw()
    plt.show()
