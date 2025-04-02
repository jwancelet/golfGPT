

# 🏌️‍♂️ GolfGPT: a Q&A Chatbot ⛳

GolfGPT: An AI-powered chatbot designed to help users learn golf. This application leverages OpenAI's language model and embeddings within a Retrieval-Augmented Generation (RAG) framework orchestrated by LangChain. It retrieves contextually relevant information from a Chroma vector database, interacts with users via a Streamlit interface, and is hosted on AWS EC2.

<h1>Features</h1>
  

1. Knowledge Base Initialization: <br>
o	Reads a predefined PDF document from the local filesystem using the PyPDF2 library.<br>
o	On the application's first run, it processes this PDF text and creates (or loads if already existing) a persistent vector database using ChromaDB, stored locally in the ./chroma_db_store directory.<br>

2. Document Processing and Embedding:<br>
o	Splits the extracted text from the PDF into manageable chunks using a custom Python function (chunk_text).<br>
o	Generates vector embeddings for each text chunk using OpenAI's text-embedding-3-small model via the official openai Python library (get_embedding function).<br>
o	Stores the text chunks and their corresponding embeddings within the local ChromaDB collection.<br>

3.  Contextual Query and Retrieval:<br>
o	Takes the user's query from the chat interface.<br>
o	Generates an embedding for the user's query using the same OpenAI embedding model.<br>
o	Performs a vector similarity search against the local ChromaDB collection (collection.query) to retrieve the most relevant text chunks from the golf guide PDF based on the query's embedding.<br>

4. Interactive Chatbot and Response Generation:<br>
o	Uses Streamlit (st.chat_message, st.chat_input, st.button, st.sidebar) to build the interactive chat user interface.<br>
o	Sends the user's original query along with the text of the retrieved chunks (as context) to OpenAI's gpt-4-turbo model via the openai library (generate_answer function).<br>
o	Uses a custom prompt that instructs the gpt-4-turbo model to answer the user's question based primarily on the provided context chunks, aiming for accurate, golf-specific responses derived from the PDF content.<br><br>


<h1>Use Cases</h1>
These use cases highlight the power of RAG chatbots to provide accurate, context-aware answers grounded in specific business data, improving efficiency, knowledge accessibility, and decision-making across various departments.<br><br>

Potential Business Use Cases for RAG Chatbots (like GolfGPT's underlying technology):

I. General Knowledge Access & Q&A:

General Document Q&A: Build chatbots that allow users to ask natural language questions about the content of specific documents (PDFs, Word docs, etc.) or entire document sets.

Enhanced Interactive FAQs: Move beyond static FAQ pages to systems where users can ask follow-up questions and get specific answers drawn directly from approved knowledge base articles or documentation.

Enterprise Knowledge Management & Semantic Search: Enable employees to perform semantic searches across vast internal repositories (wikis, shared drives, databases) and retrieve precise information or summaries from dynamically updated documents, rather than just keyword matching.

<br>
II. Internal Operations & Employee Support:

Corporate Policy & HR Document Query: Provide employees instant answers to specific questions about company benefits programs, HR policies, expense reporting rules, onboarding procedures, and other internal corporate documents.

IT Help Desk & Technical Support: Automate responses to common IT issues by allowing employees to query internal technical documentation, troubleshooting guides, and software manuals.

Sales Enablement & Product Information: Equip sales teams to quickly find and relay accurate product specifications, pricing details, competitor comparisons, relevant case studies, and marketing collateral during prospect interactions.

Financial Policy & Reporting Assistance: Allow finance teams and other employees to query internal financial policy documents, budget guidelines, reporting procedures, and potentially interpret data from specific reports.

Supply Chain & Logistics Query: Enable internal teams or partners to ask specific questions about inventory status, shipment tracking, supplier agreements, or logistics protocols based on internal systems and documents.

<br>
III. Customer-Facing & Support:

Customer Self-Service & Support: Offer chatbots that provide detailed answers to customer questions about product usage, troubleshooting, account management, and service details, drawing information from manuals, guides, and knowledge bases.

Product Usage & Customer Onboarding: Guide new customers through product setup, feature exploration, and best practices by answering their specific questions based on tutorials and user guides.

Personalized Product/Service Recommendations: (Requires integration) Answer customer queries about the best product/service for their needs by retrieving and comparing features/specifications from catalogs based on the user's stated requirements.

Field Service Technician Support: Provide field technicians with a mobile chatbot to quickly query technical manuals, repair histories, schematics, and parts information while on-site.

<br>
IV. Research, Development & Compliance:

Research & Development Acceleration: Allow R&D personnel to query internal archives of past research, experimental data, technical papers, and patents to quickly find relevant information and avoid duplicating efforts.

Legal Document Review & Contract Analysis: Assist legal teams in rapidly searching large volumes of contracts, regulations, case law, or internal legal opinions for specific clauses, precedents, or compliance details.

Compliance Monitoring & Audit Support: Help compliance officers and auditors efficiently locate specific information within policy documents, procedural guides, and potentially logs to verify adherence to regulations or internal standards.

<br>
V. Training & Development:

Interactive Corporate Training & Learning: Supplement training materials with a chatbot that can answer trainee questions based only on the specific course content, providing instant clarification and reinforcing learning.

<br>
<br>
<h1>UI Screenshots</h1>




<br>
<br>
<h1>Getting Started</h1>

Follow these steps to set up and run a Q&A Chatbot locally.

<h2>Prerequisites</h2>
Python 3.8+ installed on your machine.

Required Python packages: os, streamlit, PyPDF2, openai, chromadb and traceback # For detailed error logging

An OpenAI API Key. This key must be set as an environment variable.


Data Files:<br>
A PDF file containing any content.

A directory for storing the Chroma DB vector store data (the app will automatically create a folder named chroma_db_store if it doesn’t exist).
Note: The PDF file and API key are not included in this repository.

<br>
Setup <br>
Clone the Repository

bash <br>
git clone https://github.com/your-username/your-repository.git <br>
cd your-repository<br>
<br>
Install Dependencies

It’s recommended to use a virtual environment:

bash <br>
python -m venv venv <br>
source venv/bin/activate  # On Windows use: venv\Scripts\activate <br>
pip install -r requirements.txt <br> 
<br>
Configure Environment Variables <br>

Create a file named .env in the root directory of the project and add your OpenAI API key: <br>

dotenv <br>
OPENAI_API_KEY=your_openai_api_key_here<br>
Alternatively, you can export the environment variable in your shell:

bash <br>
export OPENAI_API_KEY=your_openai_api_key_here <br>

<br>
Add Your Data <br>

Place your PDF file in the root directory of the project.<br>

The app uses this file to extract and embed the text for answering questions.

(Optional) If you have a pre-built vector store in a folder (e.g., chroma_db_store), ensure it’s placed in the root directory. Otherwise, the app will create and populate it automatically on first run.
<br>

Running the App <br>
Once everything is set up, run the app with Streamlit:

<br>
bash <br>
streamlit run app.py

This will start the Chatbot. Open the provided local URL in your web browser to start asking questions about your document.

