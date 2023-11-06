from langchain.chat_models import ChatOpenAI

chat_model = ChatOpenAI()
print(chat_model.predict(text="你好", max_tokens=2048))
