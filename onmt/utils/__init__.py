"""Module defining various utilities."""
from .misc import aeq, use_gpu
from .report_manager import ReportMgr, build_report_manager
from .statistics import Statistics


__all__ = ["aeq", "use_gpu", "ReportMgr",
           "build_report_manager", "Statistics"]
