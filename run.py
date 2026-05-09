from langchain.chat_models import init_chat_model
from pypdf import PdfReader

def main():
    model = init_chat_model("deepseek-v4-flash")
    #resp = model.invoke("Hello")
    #print(resp)
    reader = PdfReader("docs/sample_1.pdf")
    for page in reader.pages:
        print(page.extract_text())

if __name__=="__main__":
    main()