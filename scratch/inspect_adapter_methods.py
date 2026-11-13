from private_llm_analytics.web.adapters import WebAnalyticsAdapter
import inspect

print(inspect.getsource(WebAnalyticsAdapter.load_demo_data))
print(inspect.getsource(WebAnalyticsAdapter.get_dataset_status))
