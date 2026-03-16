import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.maintenance.build_feedback_ltr_data import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
