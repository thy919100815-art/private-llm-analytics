path = 'src/private_llm_analytics/web/static/app.js'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read().replace('\r\n', '\n')

old_task_names = """const TASK_NAMES = {
  technical: "技术类",
  document: "文档类",
  analysis: "分析类",
};"""

new_task_names = """const TASK_NAMES = {
  document: "文档类",
  technical: "技术类",
  analysis: "分析类",
  coding: "代码编写",
  translation: "多语言翻译",
  summarization: "文本摘要",
  customer_support: "客服支持",
  creative_writing: "创意写作",
  data_extraction: "信息抽取",
  qa: "问答咨询",
  code_review: "代码审查",
  brainstorming: "头脑风暴",
};"""

assert old_task_names in content, "old_task_names not found in app.js"
content = content.replace(old_task_names, new_task_names, 1)

with open(path, 'w', encoding='utf-8', newline='\r\n') as f:
    f.write(content.replace('\n', '\r\n'))
print("Successfully updated TASK_NAMES in app.js")
