from __future__ import annotations

from fno_research.config import Settings
from fno_research.pipeline import ResearchPipeline


def build_pipeline(settings: Settings, sample: bool = False) -> ResearchPipeline:
    if sample:
        from fno_research.data.sample import SampleDataProvider, StaticNewsProvider

        return ResearchPipeline(settings, SampleDataProvider(), StaticNewsProvider())

    from fno_research.data.kite import KiteDataProvider
    from fno_research.data.news import RSSNewsProvider

    return ResearchPipeline(settings, KiteDataProvider(settings), RSSNewsProvider())
