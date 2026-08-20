import asyncio
from typing import TypedDict, Annotated
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import add_messages
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from src.database.models import BloodTestResultModel
from langchain_mcp_adapters.client import MultiServerMCPClient

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

class ClinicalLabChat:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0.0)
        self.mcp_client = MultiServerMCPClient({
            "mcp_server": {
                "transport": "stdio",
                "command": "python", 
                "args": ["mcp_server.py"]
            }
        })
        self.tools = []
        self.workflow = None
        self.chat_agent = None

    async def initialize(self):
        # fetch tools from MCP servers, then create the agent and workflow
        self.tools = await self.mcp_client.get_tools()
        self.chat_agent = self._create_chat_agent()
        self.workflow = self._create_workflow()

    def _create_chat_agent(self):
        agent = create_agent(
            model=self.llm,
            tools = self.tools,
            system_prompt=SystemMessage(content="You are a helpful assistant that can answer questions about medical documents. You have access to tools that can query the database for medical test results and user information such as age and sex. Use these tools to provide accurate and relevant information to the user."),
        )
        return agent
    
    async def _chat_node(self, state: ChatState):
        response = await self.chat_agent.ainvoke({
            "messages": state["messages"]
        })
        return response
    
    def _create_workflow(self):
        workflow = StateGraph(ChatState)

        workflow.add_node("chat", self._chat_node)
        workflow.set_entry_point("chat")
        workflow.add_edge("chat", END)

        return workflow.compile()
    
async def main():
    chat = ClinicalLabChat()
    await chat.initialize()
    state = {"messages": [HumanMessage(content="Hello, how did my Hemoglobin levels change over last two blood tests? How does my value compare to normal? What does this lab mean?")]}
    response = await chat.workflow.ainvoke(state)
    print(response["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())