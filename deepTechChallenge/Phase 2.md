### Phase 2 Action Plan (Step-by-Step)

Here is exactly what this phase entails and what we will do next:

1. **Download the Model:** Head to Hugging Face and download the specific `.gguf` file (e.g., `Phi-3-mini-4k-instruct-q4.gguf`).
    
2. **Compile `llama.cpp`:** Compiling this natively from source on your Ubuntu environment using standard `make` commands is the most efficient way to ensure it utilizes your specific CPU architecture.
    
3. **Boot the Local Server:** We will start the `llama.cpp` server in your terminal to expose the local API.
    
4. **Write the Integration (Antigravity IDE):** We will move into Antigravity to build the actual Next.js backend logic. This code will take the user's input, run the semantic search (Phase 1), structure the retrieved JSON into a strict prompt, and send it to your running `llama.cpp` server.
    

Would you like to start by compiling the `llama.cpp` server on your machine, or would you prefer to select and download the model file first?