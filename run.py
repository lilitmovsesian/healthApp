from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, AnyMessage
from pypdf import PdfReader
from typing import TypedDict, Dict, Annotated, Literal 
from langgraph.graph import StateGraph, END
from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from src.database.session import SessionLocal
from src.database.models import DocumentModel, BloodTestResultModel

MED_DOC_CLASSIFIER_SYS_PROMPT = """
        Classify the type of this medical document.
        Return the data in a structured format.
    """

EXTRACTOR_SYS_PROMPT = """        
        Extract the relevant measurement from this medical test result.
        Return the data in a structured format.
    """

class BloodTestMeasurement(TypedDict):
    test_name: str
    value: str
    unit: str

class BloodTestResultSchema(TypedDict):
    test_date: str
    measurements: list[BloodTestMeasurement]

class DocumentClassificationResult(TypedDict):
    document_type: Literal['blood_test', 'urine_test', 'stool_test', 'unknown']

class MedicalDocState(TypedDict):
    raw_document: str
    document_id: int
    document_type: DocumentClassificationResult

class ClinicalLabAssistant:
    def __init__(self):
        #self.llm = ChatDeepSeek(model="deepseek-v4-flash")
        #self.llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0.0)
        self.workflow = self._create_workflow()
        self.save_graph()
        self.med_doc_classifier_agent = self._create_med_doc_classifier_agent()
        self.blood_test_extractor_agent = self._create_blood_test_extractor_agent()


    def save_graph(self):
        image = self.workflow.get_graph().draw_mermaid_png()
        with open ("workflow.png", "wb") as f:
            f.write(image)

    def _create_blood_test_extractor_agent(self):
        agent = create_agent(
            #model=self.llm,
            model="gpt-5.4-mini",
            response_format=BloodTestResultSchema,
            tools = [],
            system_prompt=SystemMessage(content=EXTRACTOR_SYS_PROMPT),
        )
        return agent
    
    def _create_med_doc_classifier_agent(self):
        agent = create_agent(
            #model=self.llm,
            model="gpt-5.4-mini",
            response_format=DocumentClassificationResult,
            tools = [],
            system_prompt=SystemMessage(content=MED_DOC_CLASSIFIER_SYS_PROMPT),
        )
        return agent
        
    def _extract_blood_data(self, state: MedicalDocState) -> Dict:
        print("Extracting blood data...")
        response = self.blood_test_extractor_agent.invoke({
            "messages": [HumanMessage(content=state["raw_document"])]
        })

        if 'structured_response' in response:
            blood_data = response['structured_response']
            self._save_blood_test_results(state["document_id"], blood_data)
        else:
            blood_data = {}

        return {"blood_data": blood_data}
    

    def _save_blood_test_results(self, document_id: int, blood_data: BloodTestResultSchema):
        session = SessionLocal()
        try:
            results = []
            for measurement in blood_data.get("measurements", []):
                result = BloodTestResultModel(
                    document_id=document_id,
                    user_id=1,
                    test_date=blood_data["test_date"],
                    test_name=measurement["test_name"],
                    value=measurement["value"],
                    unit=measurement["unit"]
                )
                results.append(result)
            session.add_all(results)
            session.commit()
        finally:
            session.close()

    def _classify_med_doc(self, state: MedicalDocState) -> Dict:
        print("Classifying medical document...")

        response = self.med_doc_classifier_agent.invoke({
            "messages": [HumanMessage(content=state["raw_document"])]
        })

        if 'structured_response' in response:
            doc_type = response['structured_response'].get('document_type', 'unknown')
        else:
            doc_type = 'unknown'

        return {"document_type": doc_type}

    def _extract_urine_data(self, state: MedicalDocState) -> StateGraph:
        return state

    def _extract_stool_data(self, state: MedicalDocState) -> StateGraph:
        return state

    def _create_workflow(self) -> StateGraph:
        workflow = StateGraph(MedicalDocState)

        workflow.add_node("med_doc_classification", self._classify_med_doc)
        workflow.add_node("extract_blood_data", self._extract_blood_data)
        workflow.add_node("extract_urine_data", self._extract_urine_data)
        workflow.add_node("extract_stool_data", self._extract_stool_data)

        workflow.set_entry_point("med_doc_classification")

        workflow.add_conditional_edges("med_doc_classification", lambda state: state["document_type"], {'blood_test': "extract_blood_data", 'urine_test': "extract_urine_data", 'stool_test': "extract_stool_data", 'unknown': END})
        workflow.add_edge("extract_blood_data", END)
        workflow.add_edge("extract_urine_data", END)
        workflow.add_edge("extract_stool_data", END)

        return workflow.compile()


def add_document_to_db(document_content: str) -> int:
    session = SessionLocal()
    try:
        user_id = 1
        document = DocumentModel(user_id=user_id, content=document_content)
        session.add(document)
        session.commit()
        session.refresh(document)
        return document.id
    finally:
        session.close()


def main():
    reader = PdfReader("docs/sample_4.pdf")
    document = ''
    for page in reader.pages:
        text = page.extract_text()
        document += text

    document_id = add_document_to_db(document)

    assistant = ClinicalLabAssistant()

    final_state = assistant.workflow.invoke({
        "raw_document": document,
        "document_id": document_id,
    })
    
if __name__ == "__main__":
    main()