import ollama

def sLLM_reply(prompt, model="qwen2.5:7b"):
    """Gửi prompt đến sLLM offline qua Ollama."""
    response = ollama.chat(model=model, messages=[
        {'role': 'user', 'content': prompt}
    ])
    return response['message']['content']

if __name__ == "__main__":
    ans = sLLM_reply("Xin chào, bạn có thể hiểu tiếng Việt không?")
    print("Yuki:", ans)
    print(sLLM_reply("こんにちは、ユキ。元気ですか？"))  # Nhật
    print(sLLM_reply("Xin chào Yuki, hôm nay em thấy thế nào?"))  # Việt
    print(sLLM_reply("Hello Yuki, what are you feeling right now?"))  # Anh
