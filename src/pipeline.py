
from pathlib import Path
from config import get_config
from src.agent import Agent
from openai import OpenAI
from src.tools import Tools
from ultils.Pushover import PushOver

from src.rag_system import RAGSystem

class Pipeline:
    config = get_config()

    def __init__(self) -> None:
        config = self.config
        openrouter_url = config.get("openrouter_url")
        openrouter_open_key = config.get("openrouter_api_key")

        llm_client = OpenAI(api_key=openrouter_open_key, base_url=openrouter_url)
        notifier = PushOver(config)
        tools = Tools(notifier)
        name = config.get('name', "Elijah HAASTRUP")
        chat_model = config.get('chat_model', "gpt-4o-mini")

        self.agent = Agent(llm_client, tools, name, chat_model)
       
        default_project_root = Path(__file__).resolve().parent.parent
        project_root:Path = config.get("project_root", default_project_root)
        # rag system setup
        self.rag = RAGSystem(project_root)


    def parse_history_to_message (self, history: list):
        normalised_history = []

        for item in history:
            if not isinstance(item, dict):
                user_message, assitant_message = item
                if user_message:
                    normalised_history.append({"role": "user", "content": user_message })
                if assitant_message:
                    normalised_history.append({"role": "assistant", "content": assitant_message })
            else:
                role = item.get('role', '')
                content = item.get('content', '')
                if not isinstance(content, list):
                    return history

                contentType = content[0].get('type', 'text')
                content = content[0].get(contentType, '')
                if content:
                    normalised_history.append({"role": role, "content": content })

        return normalised_history

    def chat(self, query: str, history: list) -> str:

        #prevent too long a token
        contexts = []

        should_retrieve = self.agent.should_use_rag_with_Query(query)

        # get rag contexts
        if should_retrieve:
            print("[RAG] Using RAG for this query")
            top_k = self.config.get("top_k",5)
            rag_context = self.rag.retrieve( query, top_k= top_k )
            if rag_context:
                contexts.extend(rag_context)

        query_summary = self.agent.sumarize_long_query(query)
        user_prompt = f"Question: \n {query_summary}"
        if contexts:
            user_prompt += "\n## Retrieved Information context:\n"
            for doc in contexts:
                user_prompt += f"\n[{doc['source']}]:\n{doc['text']}\n"

        normalised_history = self.parse_history_to_message(history)
        messages =  normalised_history + [{"role": "user", "content": user_prompt}]

        # call agent
        response = self.agent.llm_call(messages)
        print(f'Message-observability(query): {query_summary}\n')
        print(f'Message-observability(response): {response}\n')

        return response