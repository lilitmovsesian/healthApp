from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from pypdf import PdfReader
from typing import TypedDict, Dict
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

class MedicalDocState(TypedDict):
    raw_document: str
    document_type: str

class ClinicalLabAssistant:
    def __init__(self):
        self.llm = init_chat_model("deepseek-v4-flash")
        self.workflow = self._create_workflow()

    def _classify_med_doc(self, state: MedicalDocState) -> Dict:
        prompt = PromptTemplate(
            input_variables=['document'], 
            template="""
                Determine the type of this medical document.
                Options: 'blood_test', 'urine_test', 'unknown'.
                Document: {document}
                Answer in one word.
                """
            )
        message = HumanMessage(
            content = prompt.format(document=state['raw_document'])
        )
        response = self.llm.invoke([message])
        valid_types = ['blood_test', 'urine_test', 'unknown']
        doc_type = response.content.strip().lower()
        if doc_type not in valid_types:
            doc_type = 'unknown'

        return {"document_type": doc_type}

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
    
    initial_state = {"raw_document": document, "document_type": ""}
    final_state = assistant.workflow.invoke(initial_state)
    
    print(final_state["document_type"])
    
if __name__ == "__main__":
    main()