from gradio.components.chatbot import Message
import tiktoken

class Guardrail:

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self.tokenizer = tiktoken.encoding_for_model(model)
        self.model_encoding = tiktoken.encoding_name_for_model(model)
    
    def get_tokens(self, text: str):
        return self.tokenizer.encode(text)

    def count_text_token (self, text: str):
        tokens = self.get_tokens(text)
        return len(tokens)

    def count_message_token(self, message: Message):
        tokens = 0
        role = message.get('role', '')
        content = message.get('content', '')
        tokens += self.count_text_token(role)
        tokens += self.count_text_token(content)

        tokens += 5 # cushion for chat formatting
        return tokens
    
    def count_messages_tokens(self, messages: list[Message], token_budget:int = 3000) -> int:
        total = 133 # constant representing the difference between prompt_token count returned from the api and the one calculated locally
        remaining_token = token_budget - total
        for msg in messages:

            message_token_count = self.count_message_token(msg) + 2

            if message_token_count <= remaining_token:
                total+= message_token_count
                remaining_token -= message_token_count
            
            role = msg.get('role', 'unknown')
            print(f"Total token count for {role} message is {message_token_count}")
        
        return total
    
    def chat_tail(self, history:list, size:int = 5):

        reversed_history = list(reversed(history))
        tail = reversed(reversed_history[:size])
        return list(tail)
    
    def trim_chat_history_to_max_prompt_tokens(self,
        messages: list[Message],
        max_prompt_tokens: int = 3000,
        preserve_system: bool = True
        ) -> list[Message]:
        """
        Keep only the most recent chat history while staying within max_prompt_tokens.

        Strategy:
        - Optionally preserve the first system message
        - Walk backward from the newest non-system messages
        - Add newer messages first until the token budget is exhausted
        """
        if not messages:
            return []

        kept: list[Message] = []
        remaining_budget = max_prompt_tokens

        # Preserve first system message if present
        start_idx = 0

        system_msg = messages[0]
        if preserve_system and system_msg.get("role", '').lower() == "system":
            kept.append(system_msg)
            system_tokens = self.count_message_token(system_msg)
            remaining_budget -= system_tokens
            start_idx = 1

        tail = messages[start_idx:]

        # Add most recent messages first
        selected_reversed: list[Message] = []
        for msg in reversed(tail):
            msg_tokens = self.count_message_token(msg)
            if msg_tokens <= remaining_budget:
                selected_reversed.append(msg)
                remaining_budget -= msg_tokens
            else:
                break
        kept.extend(reversed(selected_reversed))
        return kept

