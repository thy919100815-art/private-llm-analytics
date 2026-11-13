from private_llm_analytics.web.adapters import WebAnalyticsAdapter
adapter = WebAnalyticsAdapter("configs")
ledger = adapter.service.orchestrator.ledger
print("Before delete, ledger count:", len(ledger.get_entries()))
with ledger._connection() as conn:
    conn.execute("DELETE FROM privacy_ledger")
    conn.commit()
print("After delete, ledger count:", len(ledger.get_entries()))
print("adapter.list_releases count:", len(adapter.list_releases()))
