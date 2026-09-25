from __future__ import annotations

from fno_research.config import Settings
from fno_research.paper import PaperBook
from fno_research.pipeline import ResearchPipeline
from fno_research.store import FeatureStore

SOURCES = ("nse", "kite", "sample")


def build_market(settings: Settings, source: str | None = None):
    source = source or settings.data_source
    if source == "sample":
        from fno_research.data.sample import SampleDataProvider

        return SampleDataProvider()
    if source == "kite":
        from fno_research.data.kite import KiteDataProvider

        return KiteDataProvider(settings)
    if source == "nse":
        from fno_research.data.free import FreeWebProvider

        return FreeWebProvider()
    raise ValueError(f"Unknown data source {source!r}; choose from {SOURCES}")


def build_pipeline(settings: Settings, source: str | None = None) -> ResearchPipeline:
    source = source or settings.data_source
    market = build_market(settings, source)
    if source == "sample":
        from fno_research.data.sample import StaticNewsProvider

        news, flows = StaticNewsProvider(), market
    else:
        from fno_research.data.news import RSSNewsProvider

        news = RSSNewsProvider()
        if source == "kite":  # Kite has no FII/DII data; that always comes from NSE
            from fno_research.data.nse import NSEClient

            flows = NSEClient()
        else:
            flows = market
    return ResearchPipeline(
        settings, market, news, flows=flows,
        store=FeatureStore(settings.db_path), paper=PaperBook(settings.db_path),
    )
