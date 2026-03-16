import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.experiments.run_adaptive_ablation_eval import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
