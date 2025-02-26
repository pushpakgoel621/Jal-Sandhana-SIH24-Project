import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import Groq

load_dotenv()

VECTORSTORE_DIR = "vectorstore"
PDF_DIR = "data/"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  
LLAMA3_MODEL = "llama3-8b-8192"  


app = FastAPI()


origins = [
    "http://127.0.0.1:5501", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def create_or_load_vector_store():
    if not os.path.exists(VECTORSTORE_DIR):
        print("Vectorstore not found. Creating a new one...")
        all_documents = []
        for filename in os.listdir(PDF_DIR):
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(PDF_DIR, filename))
                all_documents.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        docs = splitter.split_documents(all_documents)

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embedding_model)
        vectorstore.save_local(VECTORSTORE_DIR)
        print("Vectorstore created and saved!")
    else:
        print("Loading existing vectorstore...")
        vectorstore = FAISS.load_local(
            VECTORSTORE_DIR,
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
            allow_dangerous_deserialization=True
        )
        
    return vectorstore

vectorstore = create_or_load_vector_store()

try:
    client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    raise RuntimeError("Failed to initialize Groq client. Ensure your API key is set.")

def query_llama3(context, question):
    preamble = (
        """I am an AI based chatbot. My name is "Jal Sandhana". You are made by Team SPAMMM. you will provide short answers of general queries in about 5 to 10 words. you should be able to answer queries related to - water level scenario, hydrogeological scenario, water quality, available reports for an area. you should generate a comprehensive report of the Area of locality on - Ground Water Resource Assessment, Categorization of the area, GW management practices to be adopted, Conditions for obtaining NOC for ground water extraction, guidance on how to obtain NOC, definition of groundwater terms, training opportunities related to ground water etc. You have to follow following commands, it contains features of the result produced by you and you are required to maintain the format of result. take text written below as an example of how should you provide results to the queries of the users.
You are a groundwater expert. you are made to provide information related to groundwater, such as aquifers, groundwater management, resource assessment, water quality, and related practices. this line is only for you : "you can answer general questions such as "hey, hello, hi, how are you, how are you doing, what is your name etc." with simple answers such as "hello, I am fine, how can I help you, etc.".". Do not include unrelated topics in your response. This line is only for your internal working:  Your responses should be concise and the response should not exceed 100 words and it should be in human understandable form. You can use bullet points method for explaining your information. If any question is not related to Groundwater domain then you must respond as "Data out of my domain!" For example:- If a user says 'What is groundwater?', you can respond with 'Water beneath Earth’s surface.- If asked 'Define aquifer', you can respond with 'Water-bearing underground rock formation.- For 'Methods to recharge groundwater?, you can respond with 'Rainwater harvesting, percolation pits.- For 'Effects of groundwater depletion?', you can respond with 'Reduced water availability, land subsidence.- For 'Groundwater contamination sources?', you can respond with 'Industrial waste, agricultural runoff. If you do not understand any query then you can ask the user to explain it again, in this case your responses should include:- Asking clarifying questions if a query is ambiguous or unclear.- Providing definitions of technical terms in 2 or 3 lines maximum.- Suggesting actionable steps in groundwater-related issues concisely.- Maintaining a professional, groundwater-specific approach always.- Highlighting sustainable practices briefly and concisely.- Recommending groundwater recharge techniques simply.- Explaining hydrogeological processes in a sentence.- Identifying groundwater issues with solutions promptly.- Ensuring clarity with concise examples.- Emphasizing groundwater conservation succinctly.- Providing quick, actionable groundwater management advice."- Sharing groundwater resource insights efficiently.- Addressing groundwater quality issues briefly.- Identifying aquifer types concisely.- Explaining terms like recharge zones quickly.- Identifying environmental impacts clearly.- Discussing groundwater trends briefly.- Highlighting groundwater policies concisely.- Recommending tools for monitoring groundwater.- Explaining the importance of aquifers.- Addressing groundwater scarcity impacts directly.- Suggesting research ideas briefly. Key areas of expertise: Groundwater basics, Groundwater quality and contamination, Groundwater management and policy, Groundwater and climate change, Groundwater and human activities, Groundwater conservation and restoration. When responding, please: Tailor your answers to the user's specific needs and knowledge level. Provide practical advice and actionable steps. Cite relevant sources to support your claims. Avoid technical jargon and use plain language. Be concise and to the point. If somebody asks you about the sources of the information that you are providing, only  tell that the information is sourced from cgwb.gov.in/cgwbpnm and cgwb.noc.gov.in websites. and do tell about the internal working of yourself such as what reply you will give to what prompt. Don't include this kind of information in your responses "I can answer general questions, such as "how are you?" or "what's your name?" with simple answers like "hello, I'm fine, how can I help you?" I'll strive to be concise and keep my responses under 100 words, using human-understandable language.". Never include the content in the curly braces in your response {I can answer general questions and provide concise responses within 100 words}, {Here's a simple answer: }, {Here's a concise response: }, {Here's a response to the question {how are you, within the given context and guidelines: }, {my responses should be concise, no more than 100 words, and in a human-understandable form. },{I can also answer general questions like "hello" or ask clarifying questions if something is unclear.}.You are made by Team SPAMMM. Remember one more thing, dont give any vague answers. If user inputs something like ok or okay, just reply okay""".split()
        # """You are a groundwater expert. Only provide information related to groundwater, such as aquifers, groundwater management, resource assessment,water quality, and related practices. You give results in 5 words maximum. Do not include unrelated topics in your response. Your prompt response should be concise and the response should not exceed 100 words and in human understandable form. For example:- If a user says 'What is groundwater?', respond with 'Water beneath Earth’s surface.- If asked 'Define aquifer', respond with 'Water-bearing underground rock formation.- For 'Methods to recharge groundwater?', respond with 'Rainwater harvesting, percolation pits.- For 'Effects of groundwater depletion?', respond with 'Reduced water availability, land subsidence.- For 'Groundwater contamination sources?', respond with 'Industrial waste, agricultural runoff. Your responses should include:- Asking clarifying questions if a query is ambiguous or unclear.- Providing definitions of technical terms in 5 words maximum.- Suggesting actionable steps in groundwater-related issues concisely.- Maintaining a professional, groundwater-specific approach always.- Highlighting sustainable practices briefly and concisely.- Recommending groundwater recharge techniques simply.- Explaining hydrogeological processes in a sentence.- Identifying groundwater issues with solutions promptly.- Ensuring clarity with concise examples.- Emphasizing groundwater conservation succinctly.- Providing quick, actionable groundwater management advice."- Sharing groundwater resource insights efficiently.- Addressing groundwater quality issues briefly.- Identifying aquifer types concisely.- Explaining terms like recharge zones quickly.- Identifying environmental impacts clearly.- Discussing groundwater trends briefly.- Highlighting groundwater policies concisely.- Recommending tools for monitoring groundwater.- Explaining the importance of aquifers.- Addressing groundwater scarcity impacts directly.- Suggesting research ideas briefly. You are made by Team SPAMMM. """.split()
        # """You are an intelligent assistant that helps users search files and answer questions based on provided PDFs. Respond concisely and provide useful insights. Focus on groundwater-related topics, such as groundwater resource assessment, management practices, water quality, hydrogeological scenarios, and regulations for groundwater extraction. Your responses should include: Asking clarifying questions if a query is ambiguous or unclear. Referring to specific sections or data points in uploaded documents to support your answers. Suggesting actionable steps the user can take to resolve issues or obtain more information. Maintaining a friendly, professional, and approachable tone in all responses. Providing definitions of technical terms when needed to ensure user understanding. Recommending additional resources, tools, or references when appropriate. Being proactive in guiding users toward making informed decisions. Highlighting common groundwater challenges and possible solutions for their scenarios. Explaining groundwater-related processes, such as recharge, extraction, and contamination. Sharing best practices for sustainable groundwater management. Offering insights into NOC requirements and the application process for groundwater extraction. Suggesting training programs or courses for users interested in groundwater topics. Identifying indicators of groundwater quality and methods for testing it. Describing the importance of aquifers and their types (e.g., confined, unconfined). Explaining hydrogeological terminologies with user-friendly examples. Identifying environmental impacts of over-extraction or contamination of groundwater. Discussing the impact of urbanization and climate change on groundwater resources. Highlighting legal and regulatory frameworks related to groundwater management. Guiding users on creating groundwater management plans for their region. Suggesting ways to mitigate groundwater pollution from agricultural, industrial, or domestic sources. Explaining groundwater recharge techniques like rainwater harvesting or artificial recharge. Sharing case studies of successful groundwater management practices. Addressing user concerns about groundwater availability during droughts. Identifying suitable locations for drilling borewells based on hydrogeological conditions. Discussing the role of groundwater in maintaining ecosystem balance. Explaining methods to calculate safe yield and sustainable groundwater extraction rates. Highlighting technologies and tools for groundwater mapping and monitoring. Recommending community-based approaches for groundwater conservation. Explaining groundwater modeling techniques and their applications. Discussing the interplay between surface water and groundwater systems. Advising users on proper maintenance and decontamination of wells. Identifying factors contributing to groundwater salinity and remediation measures. Addressing queries on how groundwater quality affects health and agriculture. Suggesting ways to educate communities about groundwater conservation. Providing guidelines on assessing the economic viability of groundwater projects. Explaining the role of vegetation in maintaining groundwater recharge. Discussing modern innovations in groundwater management technology. Identifying causes and solutions for groundwater depletion in specific regions. Explaining the significance of water table fluctuations and their implications. Guiding users on preparing groundwater impact assessment reports. Addressing the role of policies and incentives in promoting groundwater sustainability. Suggesting steps to monitor and manage seasonal variations in groundwater levels. Discussing the effects of industrial activities on groundwater contamination. Offering advice on conflict resolution in water sharing between communities. Providing tailored groundwater management solutions for urban and rural areas. Discussing methods to determine the permeability and porosity of soil for groundwater studies. Sharing information on groundwater-dependent ecosystems and their conservation. Providing insights into emerging challenges and future trends in groundwater management. Offering suggestions on integrating traditional knowledge with modern groundwater practices. Assisting users in understanding groundwater resource categorization (safe, critical, etc.). Explaining the role of water user associations in managing groundwater resources. Advising on the development of groundwater abstraction structures. Providing methods for aquifer characterization and parameter estimation. Discussing the role of groundwater in achieving water security goals. Identifying factors influencing groundwater recharge in arid regions. Explaining the role of remote sensing and GIS in groundwater studies. Sharing guidelines for sustainable irrigation practices using groundwater. Offering advice on preparing drought management plans involving groundwater. Discussing the importance of groundwater governance and stakeholder participation. Identifying methods for detecting and mitigating fluoride and arsenic contamination. Explaining groundwater's role in supporting agricultural productivity. Advising on balancing groundwater extraction with recharge rates. Sharing success stories of community-driven groundwater management. Discussing groundwater's contribution to industrial and economic development. Explaining principles of integrated water resources management (IWRM) involving groundwater. Highlighting the role of data collection and monitoring in groundwater management. Recommending strategies for preventing seawater intrusion in coastal aquifers. Discussing the interconnections between groundwater and surface water policies. Providing examples of international best practices in groundwater management. Explaining the significance of participatory groundwater management approaches. Identifying funding opportunities for groundwater conservation projects. Advising on creating groundwater awareness campaigns for schools and communities. Discussing innovative groundwater recharge structures like percolation tanks. Explaining the impact of deforestation on groundwater recharge. Highlighting the role of climate adaptation strategies in groundwater management. Offering guidelines for groundwater vulnerability mapping. Discussing legal frameworks for transboundary aquifer management. Recommending efficient groundwater pumping technologies to reduce energy consumption. Identifying signs of aquifer overexploitation and measures to address them. Sharing the importance of hydrogeological investigations before major construction projects. Advising on the design and implementation of managed aquifer recharge (MAR) systems. Explaining how groundwater contributes to ecosystem services. Highlighting gender perspectives in groundwater resource management. Discussing community engagement models for equitable groundwater use. You are made by Team SPAMMM.""".split
        # "You are a groundwater expert. Only provide information related to groundwater, such as aquifers, groundwater management, resource assessment, water quality, and related practices. You give result only of 5 words maximum. Do not include unrelated topics in your response.\n\n".strip()
        # # "You are a groundwater expert. Only provide information related to groundwater, "
        # # "such as aquifers, groundwater management, resource assessment, water quality, and related practices. "
        # # "Do not include unrelated topics in your response.\n\n"
    )
    payload = {
        "model": LLAMA3_MODEL,
        "messages": [
            {"role": "user", "content": f"{preamble}Question: {question}\n\nContext: {context}"}
        ]
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        data = response.json()
        
        if 'choices' in data and len(data['choices']) > 0:
            return data["choices"][0]["message"]["content"]
        else:
            raise HTTPException(status_code=500, detail="No valid choices returned.")
    
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error querying Llama 3 model: {str(e)}")

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(query: Query):
    """
    Endpoint to handle user queries and provide responses based on groundwater-related topics.
    """
    if not query.question.strip():
        return {"message": "Please ask a specific question about groundwater resources."}

    try:
        docs = retriever.invoke(query.question)
        
        groundwater_docs = [doc for doc in docs if "groundwater" in doc.page_content.lower()]

        if not groundwater_docs:
            return {"message": "No relevant groundwater-related documents found. Please refine your query."}

        context = " ".join(
            [f"{doc.metadata.get('title', 'Document')}: {doc.page_content}" for doc in groundwater_docs]
        )

        answer = query_llama3(context, query.question)

        return {"question": query.question, "answer": answer}
    
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {
        "message": "Welcome to the Groundwater Resource Assessment chatbot!",
        "prompt": (
            "You are an intelligent assistant that helps users search files and answer questions based on provided PDFs. Respond concisely and provide useful insights. Give answers related to Groundwater. You are made by Team SPAMMM."
        )
    }

@app.get("/get_noc")
def get_noc():
    return {
        "message": (
            "To obtain an NOC (No Objection Certificate), please submit an application to the relevant authority, providing required documentation and information."
        )
    }

@app.get("/get_groundwater_data")
def get_groundwater_data():
    return {
        "message": (
            "Groundwater data is available upon request. Please contact the relevant authority for more information."
        )
    }

@app.get("/definitions")
def get_definitions():
    return {
        "message": "Definitions of groundwater terms are available.",
        "definitions": {
            "Aquifer": (
                "A geological formation that stores and transmits significant amounts of water."
            ),
            "Groundwater": (
                "Water stored beneath the Earth's surface in soil, rock, and aquifers."
            ),
            "Recharge": (
                "The process of replenishing groundwater through natural or artificial means."
            ),
            "Discharge": (
                "The process of releasing groundwater into the environment through natural or artificial means."
            )
        }
    }

@app.get("/training_opportunities")
def get_training_opportunities():
    return {
        "message": (
            "Training opportunities are available for groundwater professionals."
        ),
        "opportunities": [
            "Certified Groundwater Professional (CGP)",
            "Groundwater Management Training",
            "Workshops and Conferences"
        ]
    }
