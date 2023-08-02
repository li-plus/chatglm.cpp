from langchain.llms import ChatGLM

llm = ChatGLM(endpoint_url="http://127.0.0.1:8000", max_token=2048, top_p=0.7, temperature=0.95, with_history=False)
print(llm.predict("你好"))
