    for message in groupchat.messages:
        if message["role"] == "assistant":
            content = message["content"].strip().upper()
            if content == "APPROVE":
                print("✅ The output is correct and approved!")
                langfuse.flush()
                return
            elif "SUGGESTION" in content:
                print(f"❌ Suggested improvement: {message['content']}")