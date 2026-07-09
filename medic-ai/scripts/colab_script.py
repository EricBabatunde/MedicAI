import os
import subprocess
import time
import urllib.request

# 1. Install localtunnel
!npm install -g localtunnel

# 2. Build llama.cpp with CUDA if not already built
if not os.path.exists('./llama.cpp/build/bin/llama-server'):
    print("🏗️ Building llama.cpp...")
    !git clone https://github.com/ggml-org/llama.cpp.git
    # !cd llama.cpp && CUDACXX=/usr/local/cuda/bin/nvcc cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j --target llama-server
    !cd llama.cpp && CUDACXX=/usr/local/cuda/bin/nvcc cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j 1 --target llama-server
else:
    print("✅ llama-server binary found. Skipping build.")

# 3. Download Phi-4-mini if not exists
if not os.path.exists('phi-4-mini.gguf'):
    print("📥 Downloading model...")
    !wget -O phi-4-mini.gguf https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF/resolve/main/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf

# 4. Clean up existing processes to prevent 'Address already in use'
!pkill llama-server
!pkill lt

print("\n🚀 Booting Llama.cpp with 8k Context & GPU Acceleration...")
# -fa enables Flash Attention for faster inference on Phi-4
subprocess.Popen([
    "./llama.cpp/build/bin/llama-server",
    "-m", "phi-4-mini.gguf",
    "--port", "8080",
    "-c", "8192",
    "-ngl", "99",
])
time.sleep(15)

print("🌐 Creating a Fresh Public Tunnel...")
# Removing the fixed subdomain lets localtunnel assign a completely new, unoccupied URL
subprocess.Popen(["lt", "--port", "8080"], stdout=open('tunnel.log', 'w'))
time.sleep(5)

# Read the newly generated URL from the log file
with open('tunnel.log', 'r') as f:
    print("\n" + "="*60)
    print("✅ YOUR NEW FRESH COLAB SERVER IS LIVE!")
    print(f"Target URL: {f.read().strip()}/v1/chat/completions")
    print("="*60 + "\n")

ip = urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip("\n")
print(f"⚠️ IMPORTANT: Use this IP for the Localtunnel bypass: {ip}")






print("🌐 Creating Public Tunnel...")
# Use a unique subdomain to avoid collisions
SUBDOMAIN = "medic-ai-tunnel"
subprocess.Popen(["lt", "--port", "8080", "-s", SUBDOMAIN], stdout=open('tunnel.log', 'w'))
time.sleep(5)

print("\n" + "="*60)
print("✅ YOUR COLAB SERVER IS LIVE!")
print(f"Target URL: https://{SUBDOMAIN}.loca.lt/v1/chat/completions")
print("="*60 + "\n")

ip = urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip("\n")
print(f"⚠️ IMPORTANT: Use this IP for the Localtunnel bypass: {ip}")







# 4. Clean up existing processes to prevent conflicts
!pkill llama-server
!pkill cloudflared
!pkill lt


# Download the official Cloudflare Tunnel binary
if not os.path.exists('cloudflared'):
    print("📥 Downloading Cloudflare Tunnel binary...")
    !wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
    !chmod +x cloudflared

print("\n🚀 Booting Llama.cpp with 8k Context & GPU Acceleration...")
# We redirect outputs to server.log to catch any silent GPU errors
with open('server.log', 'w') as f:
    subprocess.Popen([
        "./llama.cpp/build/bin/llama-server",
        "-m", "phi-4-mini.gguf",
        "--port", "8080",
        "-c", "8192",
        "-ngl", "99"
    ], stdout=f, stderr=f)

print("⏳ Waiting for model to load into VRAM and initialize...")
time.sleep(20) 

print("🌐 Creating a Robust Cloudflare Tunnel...")
# Start the free Cloudflare tunnel
with open('cloudflare.log', 'w') as f:
    subprocess.Popen(["./cloudflared", "tunnel", "--url", "http://127.0.0.1:8080"], stdout=f, stderr=f)
time.sleep(8)

# THE FIX: Use Regex to dynamically hunt down the exact URL structure
cf_url = None
with open('cloudflare.log', 'r') as f:
    logs = f.read()
    # Scans for "https://[anything].trycloudflare.com"
    match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", logs)
    if match:
        cf_url = match.group(0)

if cf_url:
    print("\n" + "="*60)
    print("✅ YOUR CLOUDFLARE GPU SERVER IS LIVE!")
    print(f"Copy this exact URL into your Next.js .env.local file:")
    print(f"{cf_url}/v1/chat/completions")
    print("="*60 + "\n")
else:
    print("\n❌ Could not find the Cloudflare URL. Let's check the logs:")
    print("="*60)
    print(logs)
    print("="*60)