from private_llm_analytics.web.adapters import WebAnalyticsAdapter
import inspect

adapter = WebAnalyticsAdapter("configs")
ledger = adapter.service.orchestrator.ledger
print("Ledger type:", type(ledger))
print("Ledger methods:", dir(ledger))
print("Ledger entries count:", len(ledger.get_entries()))
