# ═══════════════════════════════════════════════════════════════════════
#  Medic-AI — Next.js Clinical RAG Application
# ═══════════════════════════════════════════════════════════════════════

FROM node:20-alpine

WORKDIR /app

# Install dependencies first (cached layer if package.json unchanged)
COPY package.json package-lock.json* ./
RUN npm install

# Copy application source code + data directory
COPY . .

# Next.js dev server port
EXPOSE 3000

# Hot-reloading dev server — switch to `npm start` for production
CMD ["npm", "run", "dev"]
