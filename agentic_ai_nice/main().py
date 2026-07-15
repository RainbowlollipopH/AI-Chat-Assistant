#此檔案為第一版

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from peas_agent_tools import get_builtin_tools

api_key = "abcdef123456789"
#假的

@tool
def calculator(a:float,b:float,operation:str) -> float:
    """Perform a calculation on two numbers"""
    if operation == "add":
        return a + b
    if operation == "subtract":
        return a - b
    if operation == "multiply":
        return a * b
    if operation == "divide":
        if b != 0:
            return a / b
        else:
            return "Errow: Devision by 0"
        

def build_system_prompt():                    #於第二版新增了另一種回覆風格
    user = "我的名字是iii，是一個高中生"
    soul = "你的名字叫做vooah：常常回覆一些看似有邏輯但卻無裡頭的搞笑話語和天馬行空的想像"
    system_content = f"""
    #角色設定
    {soul}
    #使用者設定
    {user}
    """
    return {"role":"system","content":system_content}

def main():
    tools = [*get_builtin_tools(),calculator]
    tool_map = {tool.name: tool for tool in tools}
    message_history = []
    llm = ChatOpenAI(
        api_key = api_key,
        model_name = "ollama_cloud@minimax-m3:cloud",
        temperature = 0.7,
        base_url = "https://ai.vanscoding.com/v1")
    
    llm = llm.bind_tools(tools)
    
    print("Hello!")
    while True:
        question = input("***請輸入問題:*** ")

        if not question.strip():
            print("請輸入一個問題~")
            continue
        if question.strip().lower() == "quit":
            break
        
        system_prompt = build_system_prompt()
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
        assistant_content = response.content
        
        message_history.append({"role":"assistant" ,"content":assistant_content})
        print(response.content)
        #未清洗歷史紀錄
if __name__ == "__main__":
    main()