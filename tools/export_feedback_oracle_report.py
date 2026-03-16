import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.analysis.export_feedback_oracle_report import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
