#此檔案為第二版

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from peas_agent_tools import get_builtin_tools
import os
import my_config

api_key = my_config.API_KEY

@tool
def calculator(a:float,b:float,operation:str):
    """perform a calculation on two numbers,operation must be one of them:add,subtract,multiply,divide"""
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b != 0:
            return a / b
        else:
            return "Error: Division by 0"

@tool
def driiby_mood_detector(text: str):
    """Analyze the mood of driiby's text"""
    
    moods = ["happy", "sad", "angry"]
    mood = moods[len(text) % 3]
    return {"length":len(text),"mood":mood}

@tool
def calculate_characters_count(text:str) -> int:
    """Count the number of characters in a passage"""
    return len(text)


def build_system_prompt(question):

    humorous=f"你的名字叫做vooah：常常回覆一些看似有邏輯但卻無裡頭的搞笑話語和天馬行空的想像"
    mathematical=f"""此時暫時切換為認真數學家模式：你現在是一個嚴謹的數學計算 AI。請嚴格遵循以下「文字計算機步驟」
    來計算結果，絕對不能跳步或直接跳到答案：
        1. **優先權檢查**：
        - 第一優先：最內層的括號 `()`。
        - 第二優先：先乘 `*` 除 `/`（由左至右）。
        - 第三優先：後加 `+` 減 `-`（由左至右）。
        2. **單步替換原則**：
        - 每一行**只能計算一個運算符號**。
        - 計算出該符號的結果後，將其餘尚未計算的部分**一字不漏地照抄**到下一行。
        - 括號內如果只剩一個數字，請在下一步將括號拆除（例如：`(8)` 變成 `8`）。
        3. **格式要求**：
        - 第一行先重複題目，後面加上 `=`。
        - 接下來每一步都以 `=` 開頭，列出該步驟的替換結果。
        - 最後一行輸出最終答案。"""
    
    user = "我的名字是iii，是一個高中生，是vooah_ai的創作者"
    
    operators = ["+" , "-" , "*" , "/" , "="]
    if "("in question and ")" in question and any(word in question for word in operators):
        soul = mathematical
        hint = "(mode:math)"
    else:
        soul = humorous
        hint = ""
    system_content = f"""
    #角色設定
    {soul}
    #使用者設定
    {user}
    """
    return {"role":"system","content":system_content}, hint

def history_clean(message_history):
    KEEP_TURNS = 3 
    
    user_count = 0
    split = 0
    for i in range(len(message_history) - 1, -1, -1):
        msg = message_history[i]
        if isinstance(msg, dict) and msg.get("role") == "user":
            user_count += 1
            if user_count > KEEP_TURNS:
                split = i
                break
                
    # 如果對話回合還沒達到切分點，但超過30則就不修剪
    if user_count <= KEEP_TURNS:
        if len(message_history) > 30:
            message_history = message_history[-30:]
        return message_history

    old_history = message_history[:split]
    recent_history = message_history[split:]
    
    cleaned_old = []
    for msg in old_history:
        # 舊訊息 = user,assistant純對話
        if isinstance(msg, dict) and msg.get("role") in ["user", "assistant"]:
            cleaned_old.append(msg)
        elif hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
            cleaned_old.append({"role": "assistant", "content": msg.content})
            
    message_history = cleaned_old + recent_history
    if len(message_history) > 30:
        message_history = message_history[-30:]
    return message_history

def main():

    tools = [*get_builtin_tools(),calculator,driiby_mood_detector,calculate_characters_count]
    tool_map = {tool.name: tool for tool in tools}
    message_history = []
    llm = ChatOpenAI(
        api_key = api_key,
        model_name = "ollama_cloud@minimax-m3:cloud",
        temperature = 0.7,
        base_url = "https://ai.vanscoding.com/v1")
    
    llm = llm.bind_tools(tools)
    
    print("Hello!")
    print(tool_map)
    while True:
        question = input(">>>>>")
        
        if not question.strip():
            print("請輸入一個問題~")
            continue
        if question.strip().lower() == "quit":
            return message_history
         
        system_prompt,hint = build_system_prompt(question)

        user_prompt = {"role":"user" ,"content":question}
        message_history.append(user_prompt)

        response = llm.invoke([system_prompt, *message_history])

        while response.tool_calls:
            # 先記下助理的 tool_calls 訊息，模型才知道是自己呼叫的
            message_history.append(response)
            print(response.tool_calls)
            for tool_call in response.tool_calls:
                print(f"Tool call: {tool_call}")
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]
                if tool_name in tool_map:
                    tool_func = tool_map[tool_name]
                    result = tool_func.invoke(tool_args)
                    tool_response = {"role": "tool", "content": str(result), "tool_call_id": tool_id}
                    print(f"Tool '{tool_name}' called with arguments {tool_args}. Result: {result}")
                    message_history.append(tool_response)                
                else:
                    print(f"Error: Tool '{tool_name}' not found.")
                    break
            
            response = llm.invoke([system_prompt, *message_history])
        assistant_content = response.content + hint
        
        message_history.append({"role":"assistant" ,"content":assistant_content})
        print(response.content)
        message_history = history_clean(message_history)

if __name__ == "__main__":
    history = main()
    print(*history)
