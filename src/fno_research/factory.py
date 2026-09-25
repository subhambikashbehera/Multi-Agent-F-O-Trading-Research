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


def build_swing(settings: Settings, source: str | None = None, universe=None):
    """Swing scanner wired to its data, universe, books, news and the shared kill switch."""
    import dataclasses
    from datetime import date

    from fno_research.agents.news import STOCK_SYSTEM_PROMPT, NewsAgent
    from fno_research.swing.book import SwingBook
    from fno_research.swing.scanner import SwingScanner

    source = source or settings.data_source
    cfg = settings.swing
    if source == "sample":
        cfg = dataclasses.replace(cfg, universe="sample (synthetic DEMO stocks)")
    book, paper = SwingBook(settings.db_path), PaperBook(settings.db_path)
    ban_list = None
    if source == "sample":
        from fno_research.data.sample import StaticNewsProvider
        from fno_research.swing.data import SampleSwingData, sample_universe

        data = SampleSwingData()
        universe = universe or sample_universe()
        news = StaticNewsProvider([])
        ban_list = set
    else:  # Yahoo for bars, NSE for lot sizes, option chains and the ban list
        from fno_research.data.news import RSSNewsProvider
        from fno_research.data.nse import NSEClient
        from fno_research.swing.data import YahooSwingData
        from fno_research.swing.universe import load_universe

        nse = NSEClient()
        data = YahooSwingData(settings.db_path, nse=nse, option_min_days=cfg.option_min_days)
        universe = universe or load_universe(cfg.universe)
        news = RSSNewsProvider(feeds={}, google_news=True)
        ban_list = nse.ban_list
    news_agent = None
    if settings.has_anthropic:
        news_agent = NewsAgent(news, model=settings.anthropic_model,
                               system_prompt=STOCK_SYSTEM_PROMPT, subject="Company")

    def kill_switch() -> dict:
        today = date.today()
        return paper.check_kill_switch(settings.risk.daily_loss_limit, today,
                                       extra_pnl=book.daily_pnl(today))

    return SwingScanner(cfg, data, universe, book=book, paper=paper, news_agent=news_agent,
                        kill_switch=kill_switch, ban_list=ban_list)
