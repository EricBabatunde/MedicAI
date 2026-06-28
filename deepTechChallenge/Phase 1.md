Here is how you execute Phase 1:

### Step 1: Source the "Source of Truth"

Do not let the model rely on its internal training data for medical decisions. You need established, low-resource medical protocols.

- **What to download:** Download the **WHO IMAI (Integrated Management of Adolescent and Adult Illness) District Clinician Manual** or the **South African Triage Scale (SATS) Training Manual** PDFs. These are the gold standards for African clinical settings because they are algorithmic (e.g., _If Capillary refill > 3 sec AND weak pulse -> RED / Emergency_).
    

### Step 2: Data Preprocessing (Kill the PDFs)

Feeding raw PDFs directly into an LLM is a great way to confuse it and waste your limited context window. You need to extract the triage charts and convert them into clean, structured text.

- **What to do:** Manually (or with a script) extract the symptoms and their corresponding triage color codes (Red/Emergency, Orange/Very Urgent, Yellow/Urgent, Green/Routine).
    
- **Structure it in JSON or clean Markdown:** Group the data by system or symptom cluster. For example:
    
    JSON
    
    ```
    {
      "category": "Neurological",
      "symptoms": ["severe headache", "stiff neck", "altered consciousness"],
      "rule": "Suspect meningitis. Triage level: RED. Action: Give oxygen, protect from fall, consult clinician immediately."
    }
    ```
    

### Step 3: The Ultra-Lightweight RAG Stack

To make the RAG system search through your JSON guidelines without crashing your laptop, we need tiny tools.

- **The Embedding Model:** Do not run a separate Python server for embeddings. Instead, use **Transformers.js** inside your Next.js application. It allows you to download and run `all-MiniLM-L6-v2` (a tiny, ~90MB embedding model) directly in JavaScript.
    
- **The Vector Database:** Skip heavy databases like PostgreSQL/pgvector. For a database of just a few hundred medical rules, you can use **ChromaDB** running locally, or even simpler, store the vectors in a lightweight local JSON file and use a basic cosine-similarity mathematical function in JavaScript to find the matching rules.
    

### Step 4: The Retrieval & Injection Logic

When a nurse types "Patient has a severe headache and a stiff neck," your app does the following:

1. **Embeds the query:** Converts the nurse's text into a vector using your Transformers.js setup.
    
2. **Searches the Rules:** Compares that vector against your WHO/SATS JSON database and pulls the closest matching rule (e.g., the meningitis rule).
    
3. **Constructs the Prompt:** This is the most crucial part. You inject the retrieved rule into a strict system prompt before sending it to `llama.cpp`.