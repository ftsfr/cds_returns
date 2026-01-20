"""
Doit build file for CDS Returns pipeline.
"""

import platform
import shutil
import subprocess
import sys
from pathlib import Path

import chartbook

sys.path.insert(1, "./src/")

BASE_DIR = chartbook.env.get_project_root()
DATA_DIR = BASE_DIR / "_data"
OUTPUT_DIR = BASE_DIR / "_output"
NOTEBOOK_BUILD_DIR = OUTPUT_DIR / "_notebook_build"
OS_TYPE = "nix" if platform.system() != "Windows" else "windows"


def jupyter_execute_notebook(notebook):
    """Execute a notebook and convert to HTML."""
    subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--execute",
            "--to",
            "html",
            "--output-dir",
            str(OUTPUT_DIR),
            str(notebook),
        ],
        check=True,
    )


def jupyter_to_html(notebook):
    """Convert notebook to HTML without execution."""
    subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "html",
            "--output-dir",
            str(OUTPUT_DIR),
            str(notebook),
        ],
        check=True,
    )


def copy_notebook_to_build(notebook_path, dest_dir):
    """Copy notebook to build directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / notebook_path.name
    shutil.copy(notebook_path, dest_path)
    return dest_path


def task_config():
    """Create directories for data and output."""

    def create_dirs():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        NOTEBOOK_BUILD_DIR.mkdir(parents=True, exist_ok=True)

    return {
        "actions": [create_dirs],
        "targets": [DATA_DIR, OUTPUT_DIR, NOTEBOOK_BUILD_DIR],
        "verbosity": 2,
    }


def task_pull_fred():
    """Pull Treasury yields from FRED."""
    return {
        "actions": ["python src/pull_fred.py"],
        "file_dep": ["src/pull_fred.py"],
        "targets": [DATA_DIR / "fred.parquet"],
        "verbosity": 2,
        "task_dep": ["config"],
    }


def task_pull_yield_curve():
    """Pull Fed zero-coupon yield curve."""
    return {
        "actions": ["python src/pull_fed_yield_curve.py"],
        "file_dep": ["src/pull_fed_yield_curve.py"],
        "targets": [DATA_DIR / "fed_yield_curve.parquet"],
        "verbosity": 2,
        "task_dep": ["config"],
    }


def task_pull_cds():
    """Pull Markit CDS data from WRDS."""
    return {
        "actions": ["python src/pull_markit_cds.py"],
        "file_dep": ["src/pull_markit_cds.py"],
        "targets": [
            DATA_DIR / "markit_cds.parquet",
            DATA_DIR / "markit_red_crsp_link.parquet",
            DATA_DIR / "markit_cds_subsetted_to_crsp.parquet",
        ],
        "verbosity": 2,
        "task_dep": ["config"],
    }


def task_calc():
    """Calculate CDS returns using He-Kelly formula."""
    return {
        "actions": ["python src/calc_cds_returns.py"],
        "file_dep": [
            "src/calc_cds_returns.py",
            DATA_DIR / "markit_cds.parquet",
            DATA_DIR / "fed_yield_curve.parquet",
            DATA_DIR / "fred.parquet",
        ],
        "targets": [
            DATA_DIR / "markit_cds_returns.parquet",
            DATA_DIR / "markit_cds_contract_returns.parquet",
        ],
        "verbosity": 2,
        "task_dep": ["pull_fred", "pull_yield_curve", "pull_cds"],
    }


def task_format():
    """Create FTSFR standardized datasets."""
    return {
        "actions": ["python src/create_ftsfr_datasets.py"],
        "file_dep": [
            "src/create_ftsfr_datasets.py",
            DATA_DIR / "markit_cds_returns.parquet",
            DATA_DIR / "markit_cds_contract_returns.parquet",
        ],
        "targets": [
            DATA_DIR / "ftsfr_cds_portfolio_returns.parquet",
            DATA_DIR / "ftsfr_cds_contract_returns.parquet",
        ],
        "verbosity": 2,
        "task_dep": ["calc"],
    }


def task_run_notebooks():
    """Execute summary notebook and convert to HTML."""
    notebook_py = BASE_DIR / "src" / "summary_cds_returns_ipynb.py"
    notebook_ipynb = NOTEBOOK_BUILD_DIR / "summary_cds_returns.ipynb"

    def run_notebook():
        # Convert py to ipynb
        subprocess.run(
            ["ipynb-py-convert", str(notebook_py), str(notebook_ipynb)],
            check=True,
        )
        # Execute the notebook
        jupyter_execute_notebook(notebook_ipynb)

    return {
        "actions": [run_notebook],
        "file_dep": [
            notebook_py,
            DATA_DIR / "ftsfr_cds_portfolio_returns.parquet",
            DATA_DIR / "ftsfr_cds_contract_returns.parquet",
        ],
        "targets": [
            notebook_ipynb,
            OUTPUT_DIR / "summary_cds_returns.html",
        ],
        "verbosity": 2,
        "task_dep": ["format"],
    }


def task_generate_pipeline_site():
    """Generate chartbook documentation site."""
    return {
        "actions": ["chartbook build -f"],
        "file_dep": [
            "chartbook.toml",
            NOTEBOOK_BUILD_DIR / "summary_cds_returns.ipynb",
        ],
        "targets": [BASE_DIR / "docs" / "index.html"],
        "verbosity": 2,
        "task_dep": ["run_notebooks"],
    }
