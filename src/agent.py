from openai import OpenAI
import json

from src.guardrail import Guardrail
from src.tools import Tools

from config import get_config

config = get_config()

debug_mode = config.get("debug_mode", False)

class Agent:
    def __init__(self, llm_client: OpenAI, tools: Tools, name: str, model: str = "gpt-4o-mini") -> None:
        self.tools = tools
        self.name = name
        self.llm_client = llm_client
        self.chat_model = model
        self.guardrail = Guardrail(model)
        
    def get_system_prompt (self):
        name = self.name
        system_prompt = f"You are acting as {name}. You are answering questions on {name}'s website, \
        particularly questions related to {name}'s career, background, skills and experience. \
        Your responsibility is to represent {name} for interactions on the website as faithfully as possible. \
        You are given a summary of {name}'s background and LinkedIn profile which you can use to answer questions. \
        Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
        If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
        If the user is engaging in discussion outside of getting to know {name}, without giving answer, politely try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool."
        
        system_prompt += "Do not provide answers that are not grounded in retrieved Information, instead steer towards getting in touch via email"
        system_prompt += "Do not share personal information like email, phone number, address, instead politely steer towards getting in touch via email"

        return system_prompt

    def handle_tool_calls(self, tool_calls: list[dict]) -> list[dict]:
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            tool_fn = getattr(self.tools, tool_name, None)
            result = tool_fn(**arguments) if tool_fn else {"error": f"Unknown tool: {tool_name}"}
            print(f"[TOOL-CALL] Tool called: {tool_name}", flush=True)
            
            results.append({ "role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id })
        return results

    def llm_call(self, messages, token_budget:int = 4000 ) -> str:
        model = self.chat_model
        system_prompt = self.get_system_prompt()
        system_message = [{"role": "system", "content": system_prompt}]

        messages = system_message + messages

        # prevent chat history+query to exceed token_budget
        message_total_token = self.guardrail.count_messages_tokens(messages,token_budget)
        print(message_total_token, "message_total_token raw")
        

        #restrict message to budgeted token size
        messages = self.guardrail.trim_chat_history_to_max_prompt_tokens(messages, token_budget)

        message_total_token_trunc = self.guardrail.count_messages_tokens(messages,token_budget)
        print(message_total_token_trunc, "message_total_token truncated")
        # return "debugging"
        tools = self.tools.get_tools()
        done = False
        while not done:
            response = self.llm_client.chat.completions.create(model = model, messages= messages, tools= tools, temperature=0.5, max_completion_tokens=500, verbosity='low')
            finish_reason = response.choices[0].finish_reason
            
            if finish_reason == "tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_calls(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
                return response.choices[0].message.content
        
        return response.choices[0].message.content

    def should_use_rag_with_Query(self, message):
            query_check = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"Is this query asking for specific information about someone's background, experience, or skills? Answer only 'yes' or 'no'.\n\nQuery: {message}"}],
                temperature=0
            )
            response = query_check.choices[0].message.content.strip().lower()
            should_retrieve = "yes" in response
            
            return should_retrieve
    
    def sumarize_long_query(self, query: str, token_size: int = 100):
        token_length = self.guardrail.count_text_token(query);

        if token_length < token_size:
            return query
        
        #summarise query to 120 token
        message = f"""You are a query compressor for an AI system.
            Your task is to rewrite the input into a shorter version while preserving:
            - the original intent
            - key entities, constraints, and numbers
            - important context needed to answer correctly

            Rules:
            - Do NOT add new information
            - Do NOT explain anything
            - Do NOT answer the query
            - Keep it concise and clear
            - Prefer bullet points if helpful
            - Maximum length: 120 tokens

            Return ONLY the rewritten query.

            Input:
            {query}
            """

        summarize_query = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": message}],
                temperature=0,
                max_completion_tokens = token_size,
                verbosity= 'low')
        useage = summarize_query.usage
        summary = summarize_query.choices[0].message.content
        print(f"Summarizing original input query({token_length} tokens) to {useage.completion_tokens} tokens")
        if debug_mode:
            print(f"Summarizing original query: {query} to  summary: {summary}")
            print(f"usage data {useage}")
        
        return summary;
