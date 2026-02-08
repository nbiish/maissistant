from typing import Optional, Dict, Any, List
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.db.sqlite import SqliteDb
import os

class AgentBrain:
    def __init__(self, data_dir: str = "data/sessions"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.agents: Dict[str, Agent] = {}

    def get_agent(self, session_id: str, provider: str, api_key: str, model_name: str) -> Agent:
        # Determine Base URL and Model ID based on provider
        base_url = "https://openrouter.ai/api/v1" # Default OpenRouter
        
        if provider.lower() == "zenmux":
            base_url = "https://zenmux.ai/api/v1"
            if not model_name:
                model_name = "moonshotai/kimi-k2.5" # Default for Zenmux
        elif provider.lower() == "z.ai-code":
            base_url = "https://api.z.ai/api/coding/paas/v4"
            if not model_name:
                model_name = "glm-4.7"
        elif provider.lower() == "openrouter":
             base_url = "https://openrouter.ai/api/v1"
             if not model_name:
                 model_name = "moonshotai/kimi-k2.5"
        elif provider.lower() == "gemini":
            pass

        # Create the model instance
        model = OpenAIChat(
            id=model_name,
            api_key=api_key,
            base_url=base_url,
        )

        # Check if agent exists for this session
        if session_id in self.agents:
            agent = self.agents[session_id]
            # Update model if changed
            agent.model = model
            return agent

        # Unique DB for this session
        db_path = os.path.join(self.data_dir, f"{session_id}.db")
        storage = SqliteDb(db_file=db_path, session_table="agent_sessions")

        # Create new agent
        agent = Agent(
            model=model,
            db=storage,
            session_id=session_id,
            add_history_to_context=True,
            num_history_messages=10,
            description="You are a super smart buddy—technical, concise, and mimic my speaking style.",
            instructions=[
                "Prioritize using Zenmux and OpenRouter interfaces.",
                "If using Z.ai, use the Code Plan API SDK endpoint.",
                "Be helpful, concise, and accurate."
            ],
        )
        self.agents[session_id] = agent
        return agent

    async def chat(
        self, 
        message: str, 
        session_id: str, 
        provider: str, 
        api_key: str, 
        model_name: str,
        image_base64: Optional[str] = None,
        fallback_key: Optional[str] = None
    ) -> str:
        
        agent = self.get_agent(session_id, provider, api_key, model_name)
        
        messages = None
        if image_base64:
             if "," in image_base64:
                 url = image_base64
             else:
                 url = f"data:image/jpeg;base64,{image_base64}"
            
             messages = [
                 {
                     "role": "user",
                     "content": [
                         {"type": "text", "text": message},
                         {"type": "image_url", "image_url": {"url": url}}
                     ]
                 }
             ]
        
        try:
            if messages:
                 response: RunResponse = agent.run(messages=messages)
            else:
                response: RunResponse = agent.run(message)
            return response.content
        except Exception as e:
            print(f"Primary model failed: {e}")
            if fallback_key and provider != "z.ai-code":
                print("Attempting fallback to Z.ai GLM-4.7...")
                try:
                    # Switch to fallback agent (Z.ai)
                    # Note: This updates the agent for this session to use Z.ai
                    fallback_agent = self.get_agent(session_id, "z.ai-code", fallback_key, "glm-4.7")
                    
                    if messages:
                        response: RunResponse = fallback_agent.run(messages=messages)
                    else:
                        response: RunResponse = fallback_agent.run(message)
                    return f"[Fallback to GLM-4.7] {response.content}"
                except Exception as fallback_error:
                     print(f"Fallback model also failed: {fallback_error}")
                     raise e # Raise original error
            else:
                raise e
