from private_llm_analytics.service.analytics import AnalyticsService
s = AnalyticsService.from_config_dir("configs")
print("Cache type:", type(s.orchestrator.cache))
print("Cache methods:", dir(s.orchestrator.cache))
