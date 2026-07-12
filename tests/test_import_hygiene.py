"""Import-hygiene regression tests.

pandas and polars dominate cold `import simflux` time but back only the
DataFrame factory (`AssetData.from_dataframe`) and the Parquet analyzer
(`ParquetResultsAnalyzer`), so they are loaded lazily on first use. These
tests pin that property: a fresh interpreter importing simflux must not pull
them in, and the lazy paths must still work when exercised.
"""

import subprocess
import sys

import pytest


def _run(code: str) -> None:
    subprocess.run([sys.executable, "-c", code], check=True)


def test_import_simflux_does_not_import_pandas_or_polars():
    _run(
        "import simflux, sys; "
        "heavy = [m for m in ('pandas', 'polars') if m in sys.modules]; "
        "assert not heavy, f'import simflux eagerly imported: {heavy}'"
    )


def test_dataframe_construction_still_works_lazily():
    # Passing a DataFrame must load pandas on demand and build the portfolio.
    _run(
        "import pandas as pd\n"
        "import simflux as sf\n"
        "df = pd.DataFrame({\n"
        "    'asset_id': [1, 2],\n"
        "    'sector': ['Tech', 'Fin'],\n"
        "    'pd': [0.02, 0.03],\n"
        "    'lgd_mean': [0.5, 0.6],\n"
        "    'lgd_std': [0.1, 0.1],\n"
        "    'exposure': [1e6, 2e6],\n"
        "})\n"
        "p = sf.CreditPortfolio(assets=df)\n"
        "assert len(p.assets) == 2\n"
    )


def test_analyzer_loads_polars_on_demand(tmp_path):
    # Exercise the polars-backed analyzer end to end through interim storage.
    pytest.importorskip("polars")
    from simflux import CreditPortfolio, StorageConfig
    from simflux.core import SimulationConfig

    portfolio = CreditPortfolio.create_sample_portfolio(
        n_assets_per_sector=10,
        sectors=["A", "B"],
        config=SimulationConfig(seed=1),
    )
    out = tmp_path / "interim.parquet"
    result = portfolio.simulate(
        n_simulations=200,
        storage_config=StorageConfig(store_interim=True, output_path=str(out)),
    )
    analyzer = result["analyzer"]
    assert analyzer.count_simulations() == 200
