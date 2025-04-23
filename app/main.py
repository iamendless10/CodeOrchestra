import os
import time
from dotenv import load_dotenv
import autogen
from autogen import GroupChat, GroupChatManager
from langfuse import Langfuse
from langchain_core.messages import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory as LangMemRedisHistory


load_dotenv()

# Langfuse Setup
langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_PRIVATE_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host="https://cloud.langfuse.com"
)

# LLM Config
config_list = [{
    "model": "llama3-70b-8192",
    "api_key": os.getenv("GROQ_API_KEY"),
    "base_url": "https://api.groq.com/openai/v1",
    "api_type": "openai",
    "price": [0, 0]
}]

# LangMem + Redis Setup
def get_langmem(session_id: str):
    chat_history = LangMemRedisHistory(
        url="redis://localhost:6379",
        session_id=session_id
    )
    memory = ConversationBufferMemory(
        chat_memory=chat_history,
        return_messages=True,
        memory_key="chat_history"
    )
    return memory

def load_messages_from_langmem(memory):
    lc_messages = memory.chat_memory.messages
    autogen_messages = []
    for msg in lc_messages:
        if isinstance(msg, HumanMessage):
            autogen_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            autogen_messages.append({"role": "assistant", "content": msg.content})
    return autogen_messages

# Langfuse Span Wrapper
def wrap_agent_with_tracing(agent, trace_id):
    original_generate_reply = agent.generate_reply
    def wrapped_generate_reply(messages=None, sender=None, **kwargs):
        start_time = time.time()
        span = langfuse.span(
            name=f"{agent.name}_Execution",
            trace_id=trace_id,
            metadata={
                "input_messages": str(messages[-1]["content"])[:200] if messages else None,
                "sender": str(sender.name) if sender else None
            }
        )
        try:
            response = original_generate_reply(messages=messages, sender=sender, **kwargs)
            span.end(metadata={"duration_ms": (time.time() - start_time) * 1000, "output": str(response)[:200]})
            return response
        except Exception as e:
            span.end(status_message="Failed", level="ERROR", metadata={"error": str(e)})
            raise
    agent.generate_reply = wrapped_generate_reply
    return agent

# Main Logic
def refactor_and_optimize_code(description, test_input, expected_output):
    session_id = "autogen"
    memory = get_langmem(session_id)
    restored_messages = load_messages_from_langmem(memory)

    # Agents
    user_proxy = autogen.UserProxyAgent(
        name="User_Proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=6,
        code_execution_config=False,
    )

    code_writer = autogen.AssistantAgent(
        name="Code_Writer",
        system_message="Write code based on the user problem. Don't worry about optimization yet.",
        llm_config={"config_list": config_list}
    )

    refactor_agent = autogen.AssistantAgent(
        name="Refactor_Agent",
        system_message="Refactor the given code to improve readability, maintainability, and structure.",
        llm_config={"config_list": config_list}
    )

    optimizer_agent = autogen.AssistantAgent(
        name="Optimizer_Agent",
        system_message="Optimize the refactored code for performance. Identify slow parts and improve them.",
        llm_config={"config_list": config_list}
    )

    test_agent = autogen.AssistantAgent(
        name="Test_Agent",
        system_message=f"Run the code using input: {test_input}. Return PASSED if the output matches expected: {expected_output}, else return FAILED.",
        llm_config={"config_list": config_list}
    )

    reviewer_agent = autogen.AssistantAgent(
        name="Reviewer_Agent",
        system_message="You are the final reviewer. Ensure code is readable, fast, and passes tests. Respond only with: FINAL APPROVE or suggest next step.",
        llm_config={"config_list": config_list}
    )

    agents = [user_proxy, code_writer, refactor_agent, optimizer_agent, test_agent, reviewer_agent]

    groupchat = GroupChat(
        agents=agents,
        messages=restored_messages,
        max_round=20,
        speaker_selection_method="round_robin"
    )

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config={"config_list": config_list},
        is_termination_msg=lambda msg: msg.get("role") == "assistant" and "FINAL APPROVE" in msg.get("content", "").upper()
    )

    trace = langfuse.trace(name="CodeRefactorAndOptimize", metadata={
        "problem": description[:100],
        "test_input": test_input,
        "expected_output": expected_output
    })

    for agent in agents:
        wrap_agent_with_tracing(agent, trace.id)

    user_proxy.initiate_chat(
        recipient=manager,
        message=f"PROBLEM:\n{description}\n\nINPUT:\n{test_input}\n\nEXPECTED OUTPUT:\n{expected_output}"
    )

    # Save messages to LangMem
    for message in groupchat.messages:
        if message["role"] == "user":
            memory.chat_memory.add_user_message(message["content"])
        elif message["role"] == "assistant":
            memory.chat_memory.add_ai_message(message["content"])

    # Final Result Check
    for message in groupchat.messages:
        if message["role"] == "assistant" and "FINAL APPROVE" in message["content"].upper():
            print("✅ FINAL APPROVE: Code is optimized and verified.")
            langfuse.flush()
            return
        elif "FAILED" in message["content"].upper():
            print("❌ Test FAILED: Further revision needed.")

# Run
if __name__ == "__main__":
    refactor_and_optimize_code(
        description="Write a Python function to calculate the nth Fibonacci number using recursion.",
        test_input="10",
        expected_output="55"
    )
