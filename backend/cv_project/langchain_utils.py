from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = OllamaLLM(model="gemma:2b", temperature=0.7)

prompt = PromptTemplate.from_template("""
You're a motivational AI assistant helping a human stay focused during a deep work session.

Their current session:
- Duration: {duration} minutes
- Vibe: {vibe}
- Minutes passed: {minute}
- Cheat Events: {cheat_count}

Give a short, powerful message to keep them focused. Use a tone matching the vibe. Be concise and human.
""")

focus_message_chain = LLMChain(llm=llm, prompt=prompt)

def generate_focus_message(duration, vibe, minute, cheat_count):
    return focus_message_chain.run({
        "duration": duration,
        "vibe": vibe,
        "minute": minute,
        "cheat_count": cheat_count
    })
