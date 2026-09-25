import pytest

from fno_research.config import Settings
from fno_research.data.sample import SampleDataProvider


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    s = Settings()
    s.db_path = tmp_path / "test.db"
    s.risk.capital = 500_000
    s.risk.max_risk_pct = 1.0
    return s


@pytest.fixture
def provider():
    return SampleDataProvider()


@pytest.fixture
def chain(provider):
    return provider.option_chain("NIFTY")
