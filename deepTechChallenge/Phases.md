The project breaks down into four main engineering pillars: Clinical Logic, the Inference Engine, Application Architecture, and System Constraint Testing.

### 1. Grounding the Clinical Logic (The Rules Engine)

An LLM left to its own devices can hallucinate medical advice, which is highly dangerous. Your first task is to constrain the model's knowledge strictly to established regional or global triage protocols.

- **The Standard:** You will want to digitize rules from frameworks like the **WHO IMAI (Integrated Management of Adolescent and Adult Illness)** guidelines or the **South African Triage Scale (SATS)**.
    
- **The Mapping:** These guidelines map specific "sensor data" (temperature, heart rate, observable symptoms) to clear output states (Emergency, Priority/Urgent, Routine Queue).
    
- **Fine-Tuning vs. Prompt Engineering:** For an 8GB laptop constraint, rather than doing a full fine-tune, you can use **RAG (Retrieval-Augmented Generation)** with a local vector database, or heavily structure your system prompts with JSON schemas representing the triage flow. The model evaluates the symptoms against the injected protocol and outputs the priority level.
    

### 2. The Inference Engine & Quantization

You need a model that is smart enough to reason through medical descriptions but light enough not to crash the OS.

- **The Model:** A model like Phi-3-Mini (3.8B) or Llama 3 (8B) quantized to a **Q4_K_M** or **Q5_K_M** GGUF format is ideal. This compresses the model to around 3.5GB to 4.5GB, leaving breathing room in the 8GB RAM limit.
    
- **The Server:** You will run the model using the `llama.cpp` server. This exposes a local, OpenAI-compatible REST API (typically on `localhost:8080`). It acts as your backend processing unit, handling inference natively on the CPU without requiring a dedicated GPU.
    

### 3. Application Architecture & UI

The end-user—a nurse or community health worker—should never see a command-line interface. They need a robust, intuitive web interface.

- **The Frontend:** Developing the interface using a framework like Next.js works perfectly here. You can build a form-based UI where health workers input symptoms, which the app then structures into a strict prompt for the local LLM.
    
- **Local Data Management:** Since everything is off-grid, maintaining persistent patient records or a queue system requires local storage. You can implement a local JSON database architecture to track active triage cases, past decisions, and queue order.
    

### 4. Development & Hardware Constraint Testing

Because the challenge evaluates projects on standard 8GB laptops, performance profiling is critical.

- **The Environment:** Developing and testing this on an Ubuntu 22.04 environment will give you granular control over memory management. You can monitor exactly how much RAM the `llama.cpp` server consumes during token generation.
    
- **Optimization:** You will need to balance the context window size. If you feed the LLM a massive 10,000-token medical manual in the prompt, RAM usage will spike and inference will slow to a crawl. The key is retrieving and sending only the specific, relevant triage rules for the symptoms presented.