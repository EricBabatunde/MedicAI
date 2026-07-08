import { NextResponse } from 'next/server';
import { readFileSync } from 'fs';
import path from 'path';
import { getEmbedding, cosineSimilarity } from '@/lib/embedding';
import bm25 from 'wink-bm25-text-search';
import nlp from 'wink-nlp-utils';

// ──────────────────────────────────────────────────────────────────────
//  Clinical RAG Pipeline
//  Pass 0: LLM Domain Router → BM25 Keyword Search → Vector Re-rank
// ──────────────────────────────────────────────────────────────────────

const LLAMA_ENDPOINT = 'http://127.0.0.1:8080/v1/chat/completions';
const LLAMA_TIMEOUT_MS = 90_000; // 90s to allow cold-start model weight loading

// Hybrid score weighting: 60% BM25 + 40% Vector (optimised via alpha sweep MRR test)
const BM25_WEIGHT = 0.6;
const VECTOR_WEIGHT = 0.4;
const BM25_TOP_K = 75;
const FINAL_TOP_K = 5;

// ─────────────────────────────────────────────────────────────────────
//  Global Database Cache — loaded ONCE at module init, never re-read
// ─────────────────────────────────────────────────────────────────────
const CHUNK_DB_PATH = path.join(process.cwd(), 'data', 'all_extracted_chunks.json');

let ALL_CHUNKS;
try {
    const initStart = Date.now();
    console.log(`\n📦 [INIT] Loading chunk database from ${CHUNK_DB_PATH}...`);
    ALL_CHUNKS = JSON.parse(readFileSync(CHUNK_DB_PATH, 'utf-8'));
    console.log(`✅ [INIT] Loaded ${ALL_CHUNKS.length} chunks in ${Date.now() - initStart}ms`);

    // Log domain distribution at startup
    const domainCounts = {};
    for (const chunk of ALL_CHUNKS) {
        const d = chunk.domain_spoke || '(unknown)';
        domainCounts[d] = (domainCounts[d] || 0) + 1;
    }
    console.log(`📊 [INIT] Domain distribution:`);
    for (const [domain, count] of Object.entries(domainCounts).sort((a, b) => b[1] - a[1])) {
        console.log(`     ${domain}: ${count}`);
    }
} catch (err) {
    console.error(`❌ [INIT] Failed to load chunk database: ${err.message}`);
    ALL_CHUNKS = [];
}

// All valid domain_spoke keys
const VALID_DOMAINS = new Set([
    'general_medicine', 'minor_surgery', 'obstetrics', 'neonatology',
    'infectious_disease', 'pharmacology', 'emergency_medicine', 'orthopaedics',
    'chronic_care', 'mental_health', 'dermatology', 'nutrition',
    'anaesthesia', 'ophthalmology', 'dental', 'ent',
]);
const DEFAULT_DOMAIN = 'general_medicine';

// Minimum docs required for wink-bm25 consolidation
const BM25_MIN_DOCS = 3;


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

        let parsed;
        try {
            const cleaned = rawContent.replace(/```json\s*/gi, '').replace(/```\s*/g, '').trim();
            parsed = JSON.parse(cleaned);
        } catch (parseErr) {
            console.error(`❌ [PASS 0] JSON parse failed: ${parseErr.message}`);
            console.log(`⚠️  [PASS 0] Falling back to "${DEFAULT_DOMAIN}"`);
            return DEFAULT_DOMAIN;
        }

        const selectedDomain = parsed?.domain_spoke?.toLowerCase?.()?.trim?.() || '';

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


// ─────────────────────────────────────────────────────────────────────
//  BM25 Keyword Search (builds index on-the-fly per domain slice)
// ─────────────────────────────────────────────────────────────────────

/**
 * Builds a BM25 index over the given chunk array and searches it.
 * Returns up to `topK` results as [{chunk, bm25Score}, ...].
 *
 * wink-bm25 prep pipeline: tokenize → remove stop words → stem
 * This gives strong keyword recall for clinical terms.
 */
function bm25Search(chunks, query, topK = BM25_TOP_K) {
    if (chunks.length < BM25_MIN_DOCS) {
        console.log(`⚠️  [BM25] Only ${chunks.length} chunk(s) — below minimum ${BM25_MIN_DOCS}, returning all`);
        return chunks.map(chunk => ({ chunk, bm25Score: 1.0 }));
    }

    const engine = bm25();

    // Configure: search across the text body and hierarchical headers
    engine.defineConfig({
        fldWeights: {
            body: 1,
            topic: 2,  // Boost matches in the topic/chapter headers
        },
    });

    // NLP prep pipeline: tokenize, remove stop words, stem
    engine.definePrepTasks([
        nlp.string.tokenize0,
        nlp.tokens.removeWords,
        nlp.tokens.stem,
    ]);

    // Index each chunk — use string IDs (wink returns them as strings)
    for (let i = 0; i < chunks.length; i++) {
        const c = chunks[i];
        const topicText = [
            c.hierarchical_context?.chapter || '',
            c.hierarchical_context?.primary_topic || '',
            c.hierarchical_context?.sub_topic || '',
        ].filter(Boolean).join(' ');

        engine.addDoc({ body: c.text_content, topic: topicText }, String(i));
    }

    engine.consolidate();

    // Search — pass topK as second arg to override wink-bm25's internal default of 10
    const rawResults = engine.search(query, topK);

    // Diagnostic: log raw library output BEFORE any slicing or mapping
    console.log(`[DEBUG] Raw BM25 output array length: ${rawResults.length}`);

    // rawResults is [[id, score], ...] — already sorted by score desc
    const mapped = rawResults.slice(0, topK).map(([idStr, score]) => ({
        chunk: chunks[parseInt(idStr, 10)],
        bm25Score: score,
    }));

    // Safety: warn if very few results (pipeline still continues)
    if (mapped.length < 5) {
        console.warn(`⚠️  [BM25] Only ${mapped.length} result(s) returned — query may be too specific or domain slice too small`);
    }

    return mapped;
}


// ─────────────────────────────────────────────────────────────────────
//  Vector Re-ranking (Transformers.js cosine similarity)
// ─────────────────────────────────────────────────────────────────────

/**
 * Embeds the query and each candidate chunk's text, then re-ranks
 * by cosine similarity. Returns results with vectorScore attached.
 */
async function vectorRerank(candidates, query) {
    console.log(`🔬 [VECTOR] Embedding query + ${candidates.length} candidate chunk(s)...`);
    const embedStart = Date.now();

    // Embed the query
    const queryEmbedding = await getEmbedding(query);

    // Embed each candidate's searchable text
    const reranked = await Promise.all(
        candidates.map(async ({ chunk, bm25Score }) => {
            const searchText = [
                chunk.hierarchical_context?.chapter || '',
                chunk.hierarchical_context?.primary_topic || '',
                chunk.hierarchical_context?.sub_topic || '',
                chunk.text_content,
            ].filter(Boolean).join(' ');

            // Truncate to ~500 words to keep embedding fast
            const truncated = searchText.split(/\s+/).slice(0, 500).join(' ');
            const chunkEmbedding = await getEmbedding(truncated);
            const vectorScore = cosineSimilarity(queryEmbedding, chunkEmbedding);

            return { chunk, bm25Score, vectorScore };
        })
    );

    console.log(`✅ [VECTOR] Embedded ${candidates.length} chunks in ${Date.now() - embedStart}ms`);
    return reranked;
}


// ─────────────────────────────────────────────────────────────────────
//  Hybrid Score Fusion (normalise + weighted combine)
// ─────────────────────────────────────────────────────────────────────

/**
 * Normalises BM25 and vector scores to [0, 1] via min-max,
 * then fuses them with the configured weights.
 * Returns results sorted by hybrid_score descending.
 */
function computeHybridScores(results) {
    if (results.length === 0) return [];

    // Find min/max for normalisation
    const bm25Scores = results.map(r => r.bm25Score);
    const vectorScores = results.map(r => r.vectorScore);

    const bm25Min = Math.min(...bm25Scores);
    const bm25Max = Math.max(...bm25Scores);
    const vecMin = Math.min(...vectorScores);
    const vecMax = Math.max(...vectorScores);

    const bm25Range = bm25Max - bm25Min || 1;  // avoid division by zero
    const vecRange = vecMax - vecMin || 1;

    const scored = results.map(r => {
        const normBm25 = (r.bm25Score - bm25Min) / bm25Range;
        const normVector = (r.vectorScore - vecMin) / vecRange;
        const hybridScore = (BM25_WEIGHT * normBm25) + (VECTOR_WEIGHT * normVector);

        return { ...r, normBm25, normVector, hybridScore };
    });

    return scored.sort((a, b) => b.hybridScore - a.hybridScore);
}


// ══════════════════════════════════════════════════════════════════════
//  POST /api/search — Asymmetric Hybrid Search Pipeline
// ══════════════════════════════════════════════════════════════════════
export async function POST(req) {
    const pipelineStart = Date.now();
    console.log('\n══════════════════════════════════════════════════════════════');
    console.log('  🚀 RAG PIPELINE — Pass 0 → BM25 → Vector Hybrid Search');
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
        console.log(`📦 [DB] Global cache: ${ALL_CHUNKS.length} chunks in memory`);

        // ────────────────────────────────────────────────────────────
        // PASS 0: Domain Classification via LLM Router
        // ────────────────────────────────────────────────────────────
        const selectedDomain = await routeQueryToDomain(query);

        // ── Filter to domain slice ──────────────────────────────────
        console.log(`\n🔬 [FILTER] Slicing to domain_spoke="${selectedDomain}"...`);
        const domainSlice = ALL_CHUNKS.filter(
            chunk => chunk.domain_spoke === selectedDomain
        );
        console.log(`📊 [FILTER] Isolated ${domainSlice.length} / ${ALL_CHUNKS.length} chunks (${((domainSlice.length / ALL_CHUNKS.length) * 100).toFixed(1)}%)`);

        // Fallback to full database if domain slice is empty
        const searchPool = domainSlice.length > 0 ? domainSlice : ALL_CHUNKS;
        if (domainSlice.length === 0) {
            console.warn(`⚠️  [FILTER] Zero chunks in "${selectedDomain}" — searching full database`);
        }

        // ────────────────────────────────────────────────────────────
        // PASS 1: BM25 Keyword Search (fast filter on domain slice)
        // ────────────────────────────────────────────────────────────
        console.log(`\n── PASS 1: BM25 Keyword Search ────────────────────────────`);
        const bm25Start = Date.now();
        const bm25Results = bm25Search(searchPool, query, BM25_TOP_K);
        const bm25Ms = Date.now() - bm25Start;

        console.log(`📊 [BM25] Retrieved ${bm25Results.length} candidate(s) from ${searchPool.length} chunks (${bm25Ms}ms)`);
        if (bm25Results.length > 0) {
            console.log(`📊 [BM25] Score range: ${bm25Results[0].bm25Score.toFixed(3)} → ${bm25Results[bm25Results.length - 1].bm25Score.toFixed(3)}`);
        }

        if (bm25Results.length === 0) {
            console.warn(`⚠️  [BM25] No keyword matches found — returning empty results`);
            return NextResponse.json({
                routed_domain: selectedDomain,
                domain_chunk_count: domainSlice.length,
                bm25_candidates: 0,
                pipeline_ms: Date.now() - pipelineStart,
                results: [],
            });
        }

        // ────────────────────────────────────────────────────────────
        // PASS 2: Vector Re-ranking (semantic scalpel on BM25 top-K)
        // ────────────────────────────────────────────────────────────
        console.log(`\n── PASS 2: Vector Re-ranking (${bm25Results.length} candidates) ─────────`);
        const vectorResults = await vectorRerank(bm25Results, query);

        // ────────────────────────────────────────────────────────────
        // FUSION: Hybrid Score (40% BM25 + 60% Vector)
        // ────────────────────────────────────────────────────────────
        console.log(`\n── FUSION: Hybrid Score (${(BM25_WEIGHT * 100).toFixed(0)}% BM25 + ${(VECTOR_WEIGHT * 100).toFixed(0)}% Vector) ──`);
        const hybridResults = computeHybridScores(vectorResults);

        // Take the top 5
        const topResults = hybridResults.slice(0, FINAL_TOP_K);

        // ── Console tracing ─────────────────────────────────────────
        console.log(`\n── TOP ${topResults.length} HYBRID RESULTS ──────────────────────────────`);
        topResults.forEach((r, i) => {
            const topic = r.chunk.hierarchical_context?.primary_topic || '(untitled)';
            const chapter = r.chunk.hierarchical_context?.chapter || '';
            const source = r.chunk.source_text || '';
            console.log(`   #${i + 1} hybrid=${r.hybridScore.toFixed(4)} | bm25=${r.normBm25.toFixed(3)} vec=${r.normVector.toFixed(3)} | ${chapter} → ${topic}`);
            console.log(`        [${r.chunk.clinical_category}] source="${source}" pages=${JSON.stringify(r.chunk.page_reference || [])}`);
        });

        // ────────────────────────────────────────────────────────────
        // LLM SYNTHESIS: Generate clinical response from Top 5 chunks
        // ────────────────────────────────────────────────────────────
        console.log(`\n── LLM SYNTHESIS: Generating clinical response ──────────────`);

        // 1. Construct the Evidence Block
        const evidenceBlock = topResults.map((r, i) => {
            const source = r.chunk.source_text || 'Unknown source';
            const pages = (r.chunk.page_reference || []).join(', ') || 'N/A';
            const text = r.chunk.text_content;
            return `[Source: ${source}, Pages: ${pages}] ${text}`;
        }).join('\n\n');

        console.log(`📄 [SYNTHESIS] Evidence block constructed (${evidenceBlock.length} chars from ${topResults.length} chunks)`);

        // 2. Clinical System Prompt
        const synthesisSystemPrompt = `You are an offline clinical triage assistant operating in a resource-constrained environment.

You must base your entire response STRICTLY on the provided "Evidence Block". Do not use outside knowledge. If the answer is not in the text, explicitly state: "Insufficient clinical data to confirm".

You must cite the source inline when recommending dosages or procedures (e.g., [MSF Guidelines, pg 12]).

You must output your response STRICTLY as a valid JSON object. Do not include markdown code blocks (like \`\`\`json) wrapping the output, just the raw JSON.

Return exactly this JSON structure, using Markdown formatting for the string values:
{
  "clinical_assessment": "Brief markdown analysis of the condition",
  "treatment_plan": "Markdown bullet points of dosages and procedures",
  "critical_warnings": "Markdown text highlighting severe risks or contraindications",
  "references": ["Array of source names used"]
}`;

        // 3. User prompt with query + evidence
        const synthesisUserPrompt = `User Query: "${query}"

Evidence Block:
${evidenceBlock}`;

        // 4. Send to llama.cpp
        const synthesisPayload = {
            messages: [
                { role: 'system', content: synthesisSystemPrompt },
                { role: 'user', content: synthesisUserPrompt },
            ],
            temperature: 0.1,
            max_tokens: 800,
        };

        console.log(`🦙 [SYNTHESIS] Sending to llama.cpp (temp=0.1, max_tokens=800, timeout=${LLAMA_TIMEOUT_MS / 1000}s)...`);
        const synthesisStart = Date.now();

        const synthesisController = new AbortController();
        const synthesisTimeoutId = setTimeout(() => {
            console.error(`⏱️ [SYNTHESIS] Request timed out after ${LLAMA_TIMEOUT_MS / 1000}s — aborting`);
            synthesisController.abort();
        }, LLAMA_TIMEOUT_MS);

        let clinicalResponse;
        try {
            const llmResponse = await fetch(LLAMA_ENDPOINT, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(synthesisPayload),
                signal: synthesisController.signal,
            });

            if (!llmResponse.ok) {
                const errText = await llmResponse.text();
                throw new Error(`llama.cpp synthesis error (${llmResponse.status}): ${errText}`);
            }

            const llmData = await llmResponse.json();
            const rawContent = llmData.choices?.[0]?.message?.content ?? '';
            const synthesisMs = Date.now() - synthesisStart;
            console.log(`🦙 [SYNTHESIS] Response received (${rawContent.length} chars, ${synthesisMs}ms)`);

            // 5. Parse the LLM's JSON response
            try {
                // Strip markdown fences if the model wraps its output despite instructions
                const cleaned = rawContent
                    .replace(/```json\s*/gi, '')
                    .replace(/```\s*/g, '')
                    .trim();
                clinicalResponse = JSON.parse(cleaned);
            } catch (parseErr) {
                console.error(`❌ [SYNTHESIS] JSON parse failed: ${parseErr.message}`);
                console.log(`📝 [SYNTHESIS] Raw LLM output:\n${rawContent}`);
                // Fallback: wrap raw text as clinical_assessment
                clinicalResponse = {
                    clinical_assessment: rawContent,
                    treatment_plan: 'Unable to parse structured response from LLM.',
                    critical_warnings: 'Manual clinical review recommended.',
                    references: topResults.map(r => r.chunk.source_text || 'Unknown'),
                };
            }
        } catch (fetchErr) {
            const errorMsg = fetchErr.name === 'AbortError'
                ? `LLM synthesis timed out after ${LLAMA_TIMEOUT_MS / 1000}s`
                : fetchErr.message;
            console.error(`❌ [SYNTHESIS] ${errorMsg}`);
            // Fallback response on LLM failure
            clinicalResponse = {
                clinical_assessment: 'LLM synthesis unavailable. Review retrieved evidence manually.',
                treatment_plan: 'See retrieved chunks below for clinical guidance.',
                critical_warnings: errorMsg,
                references: topResults.map(r => r.chunk.source_text || 'Unknown'),
            };
        } finally {
            clearTimeout(synthesisTimeoutId);
        }

        // 6. Terminal logging — formatted clinical output
        const totalMs = Date.now() - pipelineStart;
        console.log(`\n── 📋 CLINICAL RESPONSE ─────────────────────────────────────`);
        console.log(JSON.stringify(clinicalResponse, null, 2));
        console.log(`\n✅ [PIPELINE COMPLETE] domain="${selectedDomain}" | bm25=${bm25Results.length} candidates | top=${topResults.length} | ${totalMs}ms`);
        console.log('══════════════════════════════════════════════════════════════\n');

        // ── Build response ──────────────────────────────────────────
        const responseResults = topResults.map(r => ({
            chunk_id: r.chunk.chunk_id,
            domain_spoke: r.chunk.domain_spoke,
            source_text: r.chunk.source_text,
            hierarchical_context: r.chunk.hierarchical_context,
            clinical_category: r.chunk.clinical_category,
            text_content: r.chunk.text_content,
            extracted_tables: r.chunk.extracted_tables,
            page_reference: r.chunk.page_reference,
            scores: {
                bm25_raw: r.bm25Score,
                bm25_normalised: r.normBm25,
                vector: r.vectorScore,
                vector_normalised: r.normVector,
                hybrid: r.hybridScore,
            },
        }));

        return NextResponse.json({
            // LLM-generated clinical response
            ...clinicalResponse,
            // Pipeline metadata
            routed_domain: selectedDomain,
            domain_chunk_count: domainSlice.length,
            total_chunk_count: ALL_CHUNKS.length,
            bm25_candidates: bm25Results.length,
            pipeline_ms: totalMs,
            // Retrieved evidence chunks
            results: responseResults,
        });

    } catch (error) {
        console.error(`\n❌ [PIPELINE ERROR] ${error.message}`);
        console.error(error.stack);
        return NextResponse.json({ error: error.message }, { status: 500 });
    }
}