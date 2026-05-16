from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, AnyMessage
from pypdf import PdfReader
from typing import TypedDict, Dict, Annotated
from langgraph.graph import StateGraph, END
from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from langgraph.graph.message import add_messages

MED_DOC_CLASSIFIER_SYS_PROMPT = """
        Determine the type of this medical document.
        Options: 'blood_test', 'urine_test', 'stool_test', 'smear_test', 'unknown'.
        Answer in one word.
    """

class MedicalDocState(TypedDict):
    raw_document: str
    document_type: str
    #messages: Annotated[list[AnyMessage], add_messages] 

class ClinicalLabAssistant:
    def __init__(self):
        #self.llm = init_chat_model("deepseek-v4-flash")
        self.llm = ChatDeepSeek(model="deepseek-v4-flash")
        self.workflow = self._create_workflow()
        self.med_doc_classifier_agent = self._create_med_doc_classifier_agent()

    def _create_med_doc_classifier_agent(self):
        agent = create_agent(
            model=self.llm,
            tools = [],
            system_prompt=SystemMessage(content=MED_DOC_CLASSIFIER_SYS_PROMPT),
        )
        return agent
        
    def _classify_med_doc(self, state: MedicalDocState) -> Dict:
        response = self.med_doc_classifier_agent.invoke({
            "messages": [
                #SystemMessage(content=MED_DOC_CLASSIFIER_SYS_PROMPT),
                HumanMessage(content=state["raw_document"])
            ]
        })
        print(response['messages'])
        valid_types = ['blood_test', 'urine_test', 'stool_test', 'smear_test', 'unknown']
        last_message = response['messages'][-1]
        doc_type = last_message.content.strip().lower()
        if doc_type not in valid_types:
            doc_type = 'unknown'

        return {"document_type": doc_type}

    def _extract_blood_data(self) -> StateGraph:
        pass

    def _create_workflow(self) -> StateGraph:
        workflow = StateGraph(MedicalDocState)

        workflow.add_node("med_doc_classification", self._classify_med_doc)

        workflow.set_entry_point("med_doc_classification")
        workflow.add_edge("med_doc_classification", END)

        return workflow.compile()

def main():
    reader = PdfReader("docs/sample_1.pdf")
    document = ''
    for page in reader.pages:
        text = page.extract_text()
        document += text

    assistant = ClinicalLabAssistant()
    
    final_state = assistant.workflow.invoke({"raw_document": document})
    
    print(final_state["document_type"])
    
if __name__ == "__main__":
    main()