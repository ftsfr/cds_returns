"""
Doit build file for CDS Returns pipeline.
"""

import os
import platform
import sys
from pathlib import Path

import chartbook

sys.path.insert(1, "./src/")

BASE_DIR = chartbook.env.get_project_root()
DATA_DIR = BASE_DIR / "_data"
OUTPUT_DIR = BASE_DIR / "_output"
OS_TYPE = "nix" if platform.system() != "Windows" else "windows"



## Helpers for handling Jupyter Notebook tasks
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"


# fmt: off
def jupyter_execute_notebook(notebook_path):
    return f"jupyter nbconvert --execute --to notebook --ClearMetadataPreprocessor.enabled=True --inplace {notebook_path}"
def jupyter_to_html(notebook_path, output_dir=OUTPUT_DIR):
    return f"jupyter nbconvert --to html --output-dir={output_dir} {notebook_path}"
# fmt: on


def mv(from_path, to_path):
    from_path = Path(from_path)
    to_path = Path(to_path)
    to_path.mkdir(parents=True, exist_ok=True)
    if OS_TYPE == "nix":
        command = f"mv {from_path} {to_path}"
    else:
        command = f"move {from_path} {to_path}"
    return command


def task_config():
    """Create directories for data and output."""
    def create_dirs():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "actions": [create_dirs],
        "targets": [DATA_DIR, OUTPUT_DIR],
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


notebook_tasks = {
    "summary_cds_returns_ipynb": {
        "path": "./src/summary_cds_returns_ipynb.py",
        "file_dep": [
            DATA_DIR / "ftsfr_cds_portfolio_returns.parquet",
            DATA_DIR / "ftsfr_cds_contract_returns.parquet",
        ],
        "targets": [],
    },
}
notebook_files = []
for notebook in notebook_tasks.keys():
    pyfile_path = Path(notebook_tasks[notebook]["path"])
    notebook_files.append(pyfile_path)


def task_run_notebooks():
    """Execute summary notebook and convert to HTML."""
    for notebook in notebook_tasks.keys():
        pyfile_path = Path(notebook_tasks[notebook]["path"])
        notebook_path = pyfile_path.with_suffix(".ipynb")
        yield {
            "name": notebook,
            "actions": [
                f"jupytext --to notebook --output {notebook_path} {pyfile_path}",
                jupyter_execute_notebook(notebook_path),
                jupyter_to_html(notebook_path),
                mv(notebook_path, OUTPUT_DIR),
            ],
            "file_dep": [
                pyfile_path,
                *notebook_tasks[notebook]["file_dep"],
            ],
            "targets": [
                OUTPUT_DIR / f"{notebook}.html",
                *notebook_tasks[notebook]["targets"],
            ],
            "clean": True,
            "task_dep": ["format"],
        }


def task_generate_charts():
    """Generate interactive HTML charts."""
    return {
        "actions": ["python src/generate_chart.py"],
        "file_dep": [
            "src/generate_chart.py",
            DATA_DIR / "ftsfr_cds_portfolio_returns.parquet",
        ],
        "targets": [
            OUTPUT_DIR / "cds_returns_replication.html",
            OUTPUT_DIR / "cds_cumulative_returns.html",
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
            *notebook_files,
            OUTPUT_DIR / "cds_returns_replication.html",
            OUTPUT_DIR / "cds_cumulative_returns.html",
        ],
        "targets": [BASE_DIR / "docs" / "index.html"],
        "verbosity": 2,
        "task_dep": ["run_notebooks", "generate_charts"],
    }
