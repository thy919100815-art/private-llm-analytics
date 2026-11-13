from private_llm_analytics.web.adapters import WebAnalyticsAdapter
import inspect

adapter = WebAnalyticsAdapter("configs")
ledger = adapter.service.orchestrator.ledger
print("Ledger db_path:", ledger.db_path)
print(inspect.getsource(ledger.get_entries))
