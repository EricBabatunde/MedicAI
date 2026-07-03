import { NextResponse } from 'next/server';
import { getEmbedding, cosineSimilarity } from '@/lib/embedding';
import medicalData from '@/data/medic_database.json';

// ──────────────────────────────────────────────────────────────────────
//  Multi-Pass Relational RAG Pipeline
//  Pass 1: Semantic Search → LLM Filter → Pass 2: Treatment Lookup → Final Generation
// ──────────────────────────────────────────────────────────────────────

const LLAMA_ENDPOINT = 'http://127.0.0.1:8080/v1/chat/completions';
const TREATMENT_TYPES = new Set(['drug_formulary', 'procedure', 'chronic_care']);

// In-memory cache for pre-computed database embeddings (computed once on first request)
let cachedDbEmbeddings = null;

/**
 * Pre-computes vector embeddings for every record in the database.
 * Combines primary_topic, clinical_signs, and indications into a single
 * searchable text string per record.
 */
async function ensureDbEmbeddings() {
    if (cachedDbEmbeddings) return cachedDbEmbeddings;

    const records = medicalData.medical_records;
    console.log(`\n📦 [INIT] Embedding ${records.length} database records into memory...`);
    const startTime = Date.now();

    cachedDbEmbeddings = await Promise.all(records.map(async (record) => {
        const searchableText = [
            record.primary_topic,
            ...(record.clinical_signs || []),
            ...(record.indications || []),
            ...(record.step_by_step_guide || []),
        ].join(' ');

        const embedding = await getEmbedding(searchableText);
        return { ...record, embedding };
    }));

    console.log(`✅ [INIT] Database embedded in ${Date.now() - startTime}ms\n`);
    return cachedDbEmbeddings;
}

/**
 * Performs a vector similarity search against the embedded database.
 * Optionally filters by record_type before ranking.
 *
 * @param {number[]} queryEmbedding - The embedded query vector.
 * @param {number} topK - Number of top results to return.
 * @param {Set<string>|null} allowedTypes - If provided, only records with matching record_type are considered.
 * @returns {object[]} The top-K scored records (without the raw embedding vectors).
 */
function vectorSearch(queryEmbedding, topK, allowedTypes = null) {
    let pool = cachedDbEmbeddings;

    if (allowedTypes) {
        pool = pool.filter(r => allowedTypes.has(r.record_type));
    }

    const scored = pool.map(record => ({
        ...record,
        score: cosineSimilarity(queryEmbedding, record.embedding),
    })).sort((a, b) => b.score - a.score);

    // Strip raw embedding vectors before returning
    return scored.slice(0, topK).map(({ embedding, ...rest }) => rest);
}

/**
 * Sends a chat completion request to the local llama.cpp server.
 *
 * @param {string} systemPrompt - The system message guiding model behaviour.
 * @param {string} userPrompt - The user message / query context.
 * @param {number} temperature - Sampling temperature (lower = more deterministic).
 * @returns {string} The model's response text content.
 */
async function llamaChat(systemPrompt, userPrompt, temperature = 0.7) {
    const payload = {
        messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: userPrompt },
        ],
        temperature,
        // Prevent the model from generating excessively long outputs
        max_tokens: 1024,
    };

    console.log(`🦙 [LLAMA] Sending request to ${LLAMA_ENDPOINT} (temp=${temperature})...`);

    const response = await fetch(LLAMA_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });

    if (!response.ok) {
        const errText = await response.text();
        throw new Error(`llama.cpp server error (${response.status}): ${errText}`);
    }

    const data = await response.json();
    const content = data.choices?.[0]?.message?.content ?? '';
    console.log(`🦙 [LLAMA] Response received (${content.length} chars)`);
    return content;
}


// ══════════════════════════════════════════════════════════════════════
//  POST /api/search — Multi-Pass Relational RAG Pipeline
// ══════════════════════════════════════════════════════════════════════
export async function POST(req) {
    const pipelineStart = Date.now();
    console.log('\n══════════════════════════════════════════════════════════════');
    console.log('  🚀 RAG PIPELINE STARTED');
    console.log('══════════════════════════════════════════════════════════════');

    try {
        // ── Parse incoming request ──────────────────────────────────
        const { query } = await req.json();
        if (!query || typeof query !== 'string' || !query.trim()) {
            return NextResponse.json({ error: 'A non-empty query string is required.' }, { status: 400 });
        }
        console.log(`📝 [INPUT] Query: "${query}"`);

        // ── Ensure database is embedded ─────────────────────────────
        await ensureDbEmbeddings();

        // ────────────────────────────────────────────────────────────
        // PASS 1: Initial Semantic Search — Top 3 matches
        // ────────────────────────────────────────────────────────────
        console.log('\n── PASS 1: Initial Semantic Search ────────────────────────');
        const queryEmbedding = await getEmbedding(query);
        const pass1Results = vectorSearch(queryEmbedding, 3);

        pass1Results.forEach((r, i) => {
            console.log(`   #${i + 1} [${r.record_type}] ${r.primary_topic} (score: ${(r.score * 100).toFixed(1)}%) id=${r.id}`);
        });

        // ────────────────────────────────────────────────────────────
        // LLM FILTER: Diagnosis Selection
        // ────────────────────────────────────────────────────────────
        console.log('\n── LLM FILTER: Diagnosis Selection ───────────────────────');

        const filterSystemPrompt = `You are a clinical logic filter for a triage system. Your task:
1. Read the patient's symptom query.
2. Read the top 3 retrieved medical records from the database (provided as JSON).
3. Apply demographic constraints (age, gender) if the patient mentions them.
4. Select the SINGLE most accurate record that matches the patient's presentation.

You MUST respond with ONLY a valid JSON object in this exact format, with no other text:
{"selected_id": "<the id of the best-matching record>", "primary_topic": "<the primary_topic of that record>"}`;

        const filterUserPrompt = `Patient Query: "${query}"

Top 3 Retrieved Records:
${JSON.stringify(pass1Results, null, 2)}`;

        const filterRawResponse = await llamaChat(filterSystemPrompt, filterUserPrompt, 0.1);
        console.log(`📋 [LLM FILTER] Raw response: ${filterRawResponse}`);

        // Parse the LLM's JSON selection — extract JSON from possible markdown fences
        let selectedDiagnosis;
        try {
            // Strip markdown code fences if the LLM wraps its response
            const jsonStr = filterRawResponse.replace(/```json\s*/gi, '').replace(/```\s*/g, '').trim();
            selectedDiagnosis = JSON.parse(jsonStr);
        } catch (parseErr) {
            console.error(`❌ [LLM FILTER] Failed to parse LLM JSON: ${parseErr.message}`);
            console.log('⚠️  [FALLBACK] Using top Pass 1 result as fallback selection.');
            selectedDiagnosis = {
                selected_id: pass1Results[0].id,
                primary_topic: pass1Results[0].primary_topic,
            };
        }

        console.log(`✅ [LLM FILTER] Selected: id="${selectedDiagnosis.selected_id}" topic="${selectedDiagnosis.primary_topic}"`);

        // Retrieve the full selected record from the Pass 1 results
        const diagnosisRecord = pass1Results.find(r => r.id === selectedDiagnosis.selected_id) || pass1Results[0];
        console.log(`📌 [DIAGNOSIS] Locked onto: "${diagnosisRecord.primary_topic}" (${diagnosisRecord.record_type})`);

        // ────────────────────────────────────────────────────────────
        // PASS 2: Relational Lookup — Treatment records
        // ────────────────────────────────────────────────────────────
        console.log('\n── PASS 2: Relational Treatment Lookup ───────────────────');

        const treatmentQueryText = selectedDiagnosis.primary_topic;
        console.log(`🔍 [PASS 2] Searching treatments for: "${treatmentQueryText}"`);
        console.log(`🔍 [PASS 2] Filtering to types: ${[...TREATMENT_TYPES].join(', ')}`);

        const treatmentEmbedding = await getEmbedding(treatmentQueryText);
        const pass2Results = vectorSearch(treatmentEmbedding, 2, TREATMENT_TYPES);

        pass2Results.forEach((r, i) => {
            console.log(`   #${i + 1} [${r.record_type}] ${r.primary_topic} (score: ${(r.score * 100).toFixed(1)}%) id=${r.id}`);
        });

        // ────────────────────────────────────────────────────────────
        // FINAL GENERATION: Synthesise clinical recommendation
        // ────────────────────────────────────────────────────────────
        console.log('\n── FINAL GENERATION: Clinical Recommendation ─────────────');

        const clinicalContext = `
=== SELECTED TRIAGE DIAGNOSIS ===
${JSON.stringify(diagnosisRecord, null, 2)}

=== RELATED TREATMENT RECORDS ===
${pass2Results.map((r, i) => `--- Treatment ${i + 1} ---\n${JSON.stringify(r, null, 2)}`).join('\n\n')}
`;

        const generationSystemPrompt = `You are a clinical decision-support assistant for healthcare workers in resource-limited settings. Given the clinical context retrieved from the knowledge base and the patient's presenting symptoms, synthesize a clear, professional clinical recommendation.

Your recommendation MUST contain these sections:
1. **Diagnosis**: The identified condition and its triage level.
2. **Immediate Actions**: Step-by-step actions to take right now, based on the triage protocol.
3. **Medication / Treatment Protocol**: Specific drug dosages, procedures, or chronic care plans from the treatment records.

Be concise, actionable, and cite specific dosages or steps from the provided context. Do not fabricate information not present in the clinical context.`;

        const generationUserPrompt = `Patient Presentation: "${query}"

Clinical Context:
${clinicalContext}`;

        const recommendation = await llamaChat(generationSystemPrompt, generationUserPrompt, 0.3);
        console.log(`📝 [GENERATION] Recommendation generated (${recommendation.length} chars)`);
        console.log(`\n✅ [PIPELINE COMPLETE] Total time: ${Date.now() - pipelineStart}ms`);
        console.log('══════════════════════════════════════════════════════════════\n');

        // ── Return the composite response ───────────────────────────
        return NextResponse.json({
            recommendation,
            diagnosis: diagnosisRecord,
            treatments: pass2Results,
            // Preserve backward-compat with the frontend's existing results renderer
            results: [diagnosisRecord, ...pass2Results],
        });

    } catch (error) {
        console.error(`\n❌ [PIPELINE ERROR] ${error.message}`);
        console.error(error.stack);
        return NextResponse.json({ error: error.message }, { status: 500 });
    }
}