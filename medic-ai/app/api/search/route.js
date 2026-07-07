import { NextResponse } from 'next/server';
import { readFile } from 'fs/promises';
import path from 'path';

// ──────────────────────────────────────────────────────────────────────
//  Clinical RAG Pipeline — Pass 0: Query Router
//  Classifies patient query into a domain_spoke via llama.cpp,
//  then filters the chunk database to that domain slice.
// ──────────────────────────────────────────────────────────────────────

const LLAMA_ENDPOINT = 'http://127.0.0.1:8080/v1/chat/completions';
const LLAMA_TIMEOUT_MS = 30_000; // 30s timeout for the fast router call
const CHUNK_DB_PATH = path.join(process.cwd(), 'data', 'all_extracted_chunks.json');

// All valid domain_spoke keys in our database
const VALID_DOMAINS = new Set([
    'general_medicine',
    'minor_surgery',
    'obstetrics',
    'neonatology',
    'infectious_disease',
    'pharmacology',
    'emergency_medicine',
    'orthopaedics',
    'chronic_care',
    'mental_health',
    'dermatology',
    'nutrition',
    'anaesthesia',
    'ophthalmology',
    'dental',
    'ent',
]);

const DEFAULT_DOMAIN = 'general_medicine';

// ─────────────────────────────────────────────────────────────────────
//  In-memory cache for the chunk database (loaded once)
// ─────────────────────────────────────────────────────────────────────
let cachedChunks = null;

async function loadChunkDatabase() {
    if (cachedChunks) {
        console.log(`📦 [DB] Using cached chunk database (${cachedChunks.length} chunks)`);
        return cachedChunks;
    }

    console.log(`📦 [DB] Loading chunk database from ${CHUNK_DB_PATH}...`);
    const startTime = Date.now();

    const raw = await readFile(CHUNK_DB_PATH, 'utf-8');
    cachedChunks = JSON.parse(raw);

    console.log(`✅ [DB] Loaded ${cachedChunks.length} chunks in ${Date.now() - startTime}ms`);

    // Log domain distribution for tracing
    const domainCounts = {};
    for (const chunk of cachedChunks) {
        const d = chunk.domain_spoke || '(unknown)';
        domainCounts[d] = (domainCounts[d] || 0) + 1;
    }
    console.log(`📊 [DB] Domain distribution:`);
    for (const [domain, count] of Object.entries(domainCounts).sort((a, b) => b[1] - a[1])) {
        console.log(`     ${domain}: ${count}`);
    }

    return cachedChunks;
}

// ─────────────────────────────────────────────────────────────────────
//  Pass 0: LLM Domain Router
// ─────────────────────────────────────────────────────────────────────

const ROUTER_SYSTEM_PROMPT = `You are a high-speed clinical routing matrix. Your ONLY job is to read a patient's symptoms and classify the case into exactly ONE clinical domain.

Valid domain keys (choose EXACTLY one):
general_medicine, minor_surgery, obstetrics, neonatology, infectious_disease, pharmacology, emergency_medicine, orthopaedics, chronic_care, mental_health, dermatology, nutrition, anaesthesia, ophthalmology, dental, ent

Rules:
- Trauma, burns, shock, poisoning, cardiac arrest → emergency_medicine
- Pregnancy, labour, delivery, eclampsia → obstetrics
- Newborn/neonatal care → neonatology
- Fractures, sprains, joint injuries → orthopaedics
- Surgical wounds, abscess drainage, suturing → minor_surgery
- Malaria, TB, HIV, hepatitis, STIs → infectious_disease
- Diabetes, hypertension, asthma, epilepsy, COPD → chronic_care
- Depression, psychosis, anxiety, substance abuse → mental_health
- Rashes, skin infections, wounds → dermatology
- Eye conditions → ophthalmology
- Ear, nose, throat → ent
- Dental/oral → dental
- Malnutrition, feeding → nutrition
- Anaesthesia/sedation → anaesthesia
- Drug dosing, formulary → pharmacology
- Everything else → general_medicine

Respond with ONLY raw JSON, no other text:
{"domain_spoke": "selected_key_here"}`;

/**
 * Dispatches a fast, low-token classification request to llama.cpp.
 * Returns the parsed domain_spoke string, or the default on failure.
 */
async function routeQueryToDomain(query) {
    const payload = {
        messages: [
            { role: 'system', content: ROUTER_SYSTEM_PROMPT },
            { role: 'user', content: query },
        ],
        temperature: 0.1,
        max_tokens: 30,
    };

    console.log(`\n🧭 [PASS 0] Dispatching domain classification (temp=0.1, max_tokens=30)...`);
    const routerStart = Date.now();

    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
        console.error(`⏱️ [PASS 0] Router timed out after ${LLAMA_TIMEOUT_MS / 1000}s — aborting`);
        controller.abort();
    }, LLAMA_TIMEOUT_MS);

    try {
        const response = await fetch(LLAMA_ENDPOINT, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal: controller.signal,
        });

        if (!response.ok) {
            const errText = await response.text();
            console.error(`❌ [PASS 0] llama.cpp error (${response.status}): ${errText}`);
            console.log(`⚠️  [PASS 0] Falling back to "${DEFAULT_DOMAIN}"`);
            return DEFAULT_DOMAIN;
        }

        const data = await response.json();
        const rawContent = data.choices?.[0]?.message?.content ?? '';
        const routerMs = Date.now() - routerStart;
        console.log(`🦙 [PASS 0] Raw LLM response (${routerMs}ms): ${rawContent}`);

        // ── Parse the JSON response ─────────────────────────────────
        let parsed;
        try {
            // Strip markdown fences if the LLM wraps its output
            const cleaned = rawContent.replace(/```json\s*/gi, '').replace(/```\s*/g, '').trim();
            parsed = JSON.parse(cleaned);
        } catch (parseErr) {
            console.error(`❌ [PASS 0] JSON parse failed: ${parseErr.message}`);
            console.log(`⚠️  [PASS 0] Falling back to "${DEFAULT_DOMAIN}"`);
            return DEFAULT_DOMAIN;
        }

        const selectedDomain = parsed?.domain_spoke?.toLowerCase?.()?.trim?.() || '';

        // ── Validate against known domain keys ──────────────────────
        if (VALID_DOMAINS.has(selectedDomain)) {
            console.log(`✅ [PASS 0] Domain classified: "${selectedDomain}" (${routerMs}ms)`);
            return selectedDomain;
        } else {
            console.warn(`⚠️  [PASS 0] Invalid domain "${selectedDomain}" — not in VALID_DOMAINS set`);
            console.log(`⚠️  [PASS 0] Falling back to "${DEFAULT_DOMAIN}"`);
            return DEFAULT_DOMAIN;
        }
    } catch (err) {
        if (err.name === 'AbortError') {
            console.error(`⏱️ [PASS 0] Request aborted (timeout)`);
        } else {
            console.error(`❌ [PASS 0] Fetch error: ${err.message}`);
        }
        console.log(`⚠️  [PASS 0] Falling back to "${DEFAULT_DOMAIN}"`);
        return DEFAULT_DOMAIN;
    } finally {
        clearTimeout(timeoutId);
    }
}


// ══════════════════════════════════════════════════════════════════════
//  POST /api/search — Clinical RAG Pipeline (Pass 0 Build Phase)
// ══════════════════════════════════════════════════════════════════════
export async function POST(req) {
    const pipelineStart = Date.now();
    console.log('\n══════════════════════════════════════════════════════════════');
    console.log('  🚀 RAG PIPELINE — PASS 0 QUERY ROUTER');
    console.log('══════════════════════════════════════════════════════════════');

    try {
        // ── Parse incoming request ──────────────────────────────────
        const { query } = await req.json();
        if (!query || typeof query !== 'string' || !query.trim()) {
            return NextResponse.json(
                { error: 'A non-empty query string is required.' },
                { status: 400 }
            );
        }
        console.log(`📝 [INPUT] Query: "${query}"`);

        // ── Load chunk database ─────────────────────────────────────
        const allChunks = await loadChunkDatabase();

        // ────────────────────────────────────────────────────────────
        // PASS 0: Domain Classification via LLM Router
        // ────────────────────────────────────────────────────────────
        const selectedDomain = await routeQueryToDomain(query);

        // ── Filter database to the selected domain slice ────────────
        console.log(`\n🔬 [FILTER] Slicing database to domain_spoke="${selectedDomain}"...`);
        const domainSlice = allChunks.filter(
            chunk => chunk.domain_spoke === selectedDomain
        );
        console.log(`📊 [FILTER] Isolated ${domainSlice.length} chunks out of ${allChunks.length} total`);
        console.log(`📊 [FILTER] Slice ratio: ${((domainSlice.length / allChunks.length) * 100).toFixed(1)}% of database`);

        if (domainSlice.length === 0) {
            console.warn(`⚠️  [FILTER] Zero chunks in domain "${selectedDomain}" — expanding to full database`);
        }

        // ── Temporary pipeline finish: return first 3 chunks ────────
        const resultPool = domainSlice.length > 0 ? domainSlice : allChunks;
        const previewChunks = resultPool.slice(0, 3).map(chunk => ({
            chunk_id: chunk.chunk_id,
            domain_spoke: chunk.domain_spoke,
            source_text: chunk.source_text,
            hierarchical_context: chunk.hierarchical_context,
            clinical_category: chunk.clinical_category,
            // Truncate text_content for preview to keep response lean
            text_content: chunk.text_content.length > 500
                ? chunk.text_content.substring(0, 500) + '...'
                : chunk.text_content,
            extracted_tables: chunk.extracted_tables,
            page_reference: chunk.page_reference,
        }));

        console.log(`\n── PREVIEW: Returning first ${previewChunks.length} chunk(s) from "${selectedDomain}" ──`);
        previewChunks.forEach((c, i) => {
            const topic = c.hierarchical_context?.primary_topic || '(untitled)';
            const chapter = c.hierarchical_context?.chapter || '(no chapter)';
            console.log(`   #${i + 1} [${c.clinical_category}] ${chapter} → ${topic}`);
            console.log(`        source: ${c.source_text}`);
            console.log(`        text: ${c.text_content.substring(0, 80)}...`);
        });

        const totalMs = Date.now() - pipelineStart;
        console.log(`\n✅ [PIPELINE COMPLETE] Domain="${selectedDomain}" | ${domainSlice.length} chunks isolated | ${totalMs}ms total`);
        console.log('══════════════════════════════════════════════════════════════\n');

        // ── Return temporary response ───────────────────────────────
        return NextResponse.json({
            // Pipeline metadata
            routed_domain: selectedDomain,
            domain_chunk_count: domainSlice.length,
            total_chunk_count: allChunks.length,
            pipeline_ms: totalMs,
            // Preview data
            results: previewChunks,
        });

    } catch (error) {
        console.error(`\n❌ [PIPELINE ERROR] ${error.message}`);
        console.error(error.stack);
        return NextResponse.json({ error: error.message }, { status: 500 });
    }
}