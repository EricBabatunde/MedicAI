#!/usr/bin/env node
/**
 * ═══════════════════════════════════════════════════════════════════════
 *  Alpha Sweep Test — BM25 / Vector Hybrid Ratio Optimisation
 *  ───────────────────────────────────────────────────────────────────
 *  Sweeps alpha (Vector weight) from 0.0 → 1.0 in 0.05 steps.
 *  For each alpha, runs all 16 domain test cases through:
 *    1. Domain filter → BM25 (top 75) → Vector re-rank → Hybrid fusion
 *    2. Checks if target_phrase appears in Top 5 → MRR score
 *  Prints a sorted leaderboard of Average MRR per alpha.
 *
 *  Usage:  node scripts/alpha_sweep_test.js
 * ═══════════════════════════════════════════════════════════════════════
 */

const fs = require('fs');
const path = require('path');
const bm25 = require('wink-bm25-text-search');
const nlp = require('wink-nlp-utils');
const testCases = require('./testCases');

// ─────────────────────────────────────────────────────────────────────
//  Config
// ─────────────────────────────────────────────────────────────────────
const BM25_TOP_K = 75;
const FINAL_TOP_K = 5;
const BM25_MIN_DOCS = 3;
const ALPHA_STEP = 0.05;
const DB_PATH = path.join(__dirname, '..', 'data', 'all_extracted_chunks.json');

// MRR scores by rank position (0-indexed)
const MRR_SCORES = [1.0, 0.5, 1 / 3, 0.25, 0.2];


// ─────────────────────────────────────────────────────────────────────
//  Embedding (inline — avoids ESM/CJS mismatch with lib/embedding.js)
// ─────────────────────────────────────────────────────────────────────
let extractor = null;

async function initEmbedding() {
    if (extractor) return;
    console.log('🔧 [INIT] Loading Transformers.js model (Xenova/all-MiniLM-L6-v2)...');
    const startMs = Date.now();
    // Dynamic import for ESM-only @huggingface/transformers
    const { pipeline } = await import('@huggingface/transformers');
    extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    console.log(`✅ [INIT] Model loaded in ${Date.now() - startMs}ms\n`);
}

async function getEmbedding(text) {
    const output = await extractor(text, { pooling: 'mean', normalize: true });
    return Array.from(output.data);
}

function cosineSimilarity(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
    return sum;
}


// ─────────────────────────────────────────────────────────────────────
//  BM25 Search (same logic as route.js)
// ─────────────────────────────────────────────────────────────────────
function runBm25(chunks, query, topK = BM25_TOP_K) {
    if (chunks.length < BM25_MIN_DOCS) {
        return chunks.map(chunk => ({ chunk, bm25Score: 1.0 }));
    }

    const engine = bm25();
    engine.defineConfig({
        fldWeights: { body: 1, topic: 2 },
    });
    engine.definePrepTasks([
        nlp.string.tokenize0,
        nlp.tokens.removeWords,
        nlp.tokens.stem,
    ]);

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

    const rawResults = engine.search(query, topK);

    return rawResults.map(([idStr, score]) => ({
        chunk: chunks[parseInt(idStr, 10)],
        bm25Score: score,
    }));
}


// ─────────────────────────────────────────────────────────────────────
//  Min-Max Normalisation
// ─────────────────────────────────────────────────────────────────────
function normalise(values) {
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    return values.map(v => (v - min) / range);
}


// ─────────────────────────────────────────────────────────────────────
//  Run a single test case at a given alpha
// ─────────────────────────────────────────────────────────────────────
async function runTestCase(testCase, domainSlice, alpha) {
    const { query, target_phrase } = testCase;
    const targetLower = target_phrase.toLowerCase();

    // 1. BM25 keyword search on domain slice
    const bm25Results = runBm25(domainSlice, query, BM25_TOP_K);

    if (bm25Results.length === 0) {
        return 0; // no results → MRR = 0
    }

    // 2. Vector re-rank all BM25 candidates
    const queryEmbedding = await getEmbedding(query);

    const withVector = await Promise.all(
        bm25Results.map(async ({ chunk, bm25Score }) => {
            const searchText = [
                chunk.hierarchical_context?.chapter || '',
                chunk.hierarchical_context?.primary_topic || '',
                chunk.hierarchical_context?.sub_topic || '',
                chunk.text_content,
            ].filter(Boolean).join(' ');

            // Truncate to ~500 words for embedding speed
            const truncated = searchText.split(/\s+/).slice(0, 500).join(' ');
            const chunkEmbed = await getEmbedding(truncated);
            const vectorScore = cosineSimilarity(queryEmbedding, chunkEmbed);

            return { chunk, bm25Score, vectorScore };
        })
    );

    // 3. Normalise scores (0 to 1)
    const normBm25 = normalise(withVector.map(r => r.bm25Score));
    const normVector = normalise(withVector.map(r => r.vectorScore));

    // 4. Hybrid fusion
    const hybridScored = withVector.map((r, i) => ({
        chunk: r.chunk,
        hybridScore: (alpha * normVector[i]) + ((1 - alpha) * normBm25[i]),
    }));

    // 5. Sort and slice top 5
    hybridScored.sort((a, b) => b.hybridScore - a.hybridScore);
    const top5 = hybridScored.slice(0, FINAL_TOP_K);

    // 6. MRR: find first rank where target_phrase appears
    for (let rank = 0; rank < top5.length; rank++) {
        if (top5[rank].chunk.text_content.toLowerCase().includes(targetLower)) {
            return MRR_SCORES[rank];
        }
    }

    return 0; // target not found in top 5
}


// ─────────────────────────────────────────────────────────────────────
//  Main Sweep
// ─────────────────────────────────────────────────────────────────────
async function main() {
    console.log('═══════════════════════════════════════════════════════════════');
    console.log('  🧪 ALPHA SWEEP TEST — BM25/Vector Hybrid Ratio Optimisation');
    console.log('═══════════════════════════════════════════════════════════════\n');

    // Load database
    console.log(`📦 Loading database from ${DB_PATH}...`);
    const loadStart = Date.now();
    const allChunks = JSON.parse(fs.readFileSync(DB_PATH, 'utf-8'));
    console.log(`✅ Loaded ${allChunks.length} chunks in ${Date.now() - loadStart}ms`);

    // Pre-filter domain slices (avoid re-filtering 16×21 times)
    console.log('📊 Pre-filtering domain slices...');
    const domainSlices = {};
    for (const tc of testCases) {
        if (!domainSlices[tc.domain]) {
            const slice = allChunks.filter(c => c.domain_spoke === tc.domain);
            domainSlices[tc.domain] = slice;
            console.log(`     ${tc.domain}: ${slice.length} chunks`);
        }
    }

    // Initialise embedding model (one-time cost)
    await initEmbedding();

    // Pre-compute embeddings for BM25 candidates to avoid redundant work
    // Strategy: run BM25 once per test case (results are alpha-independent),
    // embed those candidates once, then re-use across all 21 alpha steps.

    console.log('────────────────────────────────────────────────────────────');
    console.log('  Phase 1: BM25 retrieval + Vector embedding (one-time)');
    console.log('────────────────────────────────────────────────────────────\n');

    const precomputed = []; // [{testCase, bm25Results, normBm25, normVector}]

    for (let t = 0; t < testCases.length; t++) {
        const tc = testCases[t];
        const domainSlice = domainSlices[tc.domain];

        process.stdout.write(`  [${t + 1}/${testCases.length}] ${tc.domain} — "${tc.query.substring(0, 50)}..." `);

        // BM25
        const bm25Results = runBm25(domainSlice, tc.query, BM25_TOP_K);

        if (bm25Results.length === 0) {
            console.log(`→ 0 BM25 results (skipped)`);
            precomputed.push({ testCase: tc, candidates: [] });
            continue;
        }

        // Vector embeddings for all BM25 candidates
        const queryEmbedding = await getEmbedding(tc.query);

        const candidates = await Promise.all(
            bm25Results.map(async ({ chunk, bm25Score }) => {
                const searchText = [
                    chunk.hierarchical_context?.chapter || '',
                    chunk.hierarchical_context?.primary_topic || '',
                    chunk.hierarchical_context?.sub_topic || '',
                    chunk.text_content,
                ].filter(Boolean).join(' ');

                const truncated = searchText.split(/\s+/).slice(0, 500).join(' ');
                const chunkEmbed = await getEmbedding(truncated);
                const vectorScore = cosineSimilarity(queryEmbedding, chunkEmbed);

                return { chunk, bm25Score, vectorScore };
            })
        );

        // Normalise both score sets
        const bm25Scores = candidates.map(r => r.bm25Score);
        const vectorScores = candidates.map(r => r.vectorScore);
        const normBm25 = normalise(bm25Scores);
        const normVector = normalise(vectorScores);

        precomputed.push({ testCase: tc, candidates, normBm25, normVector });
        console.log(`→ ${bm25Results.length} BM25 candidates embedded ✅`);
    }

    // ──────────────────────────────────────────────────────────────
    //  Phase 2: Alpha sweep (pure math — instant)
    // ──────────────────────────────────────────────────────────────
    console.log('\n────────────────────────────────────────────────────────────');
    console.log('  Phase 2: Alpha sweep (0.00 → 1.00, step 0.05)');
    console.log('────────────────────────────────────────────────────────────\n');

    const results = []; // [{alpha, avgMrr, caseDetails}]

    for (let alphaStep = 0; alphaStep <= 20; alphaStep++) {
        const alpha = alphaStep * ALPHA_STEP;
        let totalMrr = 0;
        const caseDetails = [];

        for (const { testCase, candidates, normBm25, normVector } of precomputed) {
            if (candidates.length === 0) {
                caseDetails.push({ domain: testCase.domain, mrr: 0, rank: -1 });
                continue;
            }

            // Fuse scores at this alpha
            const hybridScored = candidates.map((r, i) => ({
                chunk: r.chunk,
                hybridScore: (alpha * normVector[i]) + ((1 - alpha) * normBm25[i]),
            }));

            hybridScored.sort((a, b) => b.hybridScore - a.hybridScore);
            const top5 = hybridScored.slice(0, FINAL_TOP_K);

            // MRR check
            const targetLower = testCase.target_phrase.toLowerCase();
            let mrr = 0;
            let foundRank = -1;
            for (let rank = 0; rank < top5.length; rank++) {
                if (top5[rank].chunk.text_content.toLowerCase().includes(targetLower)) {
                    mrr = MRR_SCORES[rank];
                    foundRank = rank + 1;
                    break;
                }
            }

            totalMrr += mrr;
            caseDetails.push({ domain: testCase.domain, mrr, rank: foundRank });
        }

        const avgMrr = totalMrr / testCases.length;
        results.push({ alpha, avgMrr, caseDetails });

        const bm25Pct = ((1 - alpha) * 100).toFixed(0);
        const vecPct = (alpha * 100).toFixed(0);
        const bar = '█'.repeat(Math.round(avgMrr * 30));
        process.stdout.write(`  α=${alpha.toFixed(2)} (BM25 ${bm25Pct}% / Vec ${vecPct}%) → MRR=${avgMrr.toFixed(4)} ${bar}\n`);
    }

    // ──────────────────────────────────────────────────────────────
    //  Leaderboard (sorted by Average MRR descending)
    // ──────────────────────────────────────────────────────────────
    results.sort((a, b) => b.avgMrr - a.avgMrr);

    console.log('\n');
    console.log('═══════════════════════════════════════════════════════════════');
    console.log('  🏆 LEADERBOARD — Sorted by Average MRR (highest first)');
    console.log('═══════════════════════════════════════════════════════════════');
    console.log('');
    console.log('  Rank │ Alpha │ BM25 %  │ Vector % │ Avg MRR  │ Visual');
    console.log('  ─────┼───────┼─────────┼──────────┼──────────┼────────────────────────────');

    results.forEach((r, i) => {
        const rank = String(i + 1).padStart(4);
        const alpha = r.alpha.toFixed(2);
        const bm25Pct = ((1 - r.alpha) * 100).toFixed(0).padStart(5);
        const vecPct = (r.alpha * 100).toFixed(0).padStart(6);
        const mrr = r.avgMrr.toFixed(4);
        const bar = '█'.repeat(Math.round(r.avgMrr * 25));
        const medal = i === 0 ? ' 🥇' : i === 1 ? ' 🥈' : i === 2 ? ' 🥉' : '';
        console.log(`  ${rank} │ ${alpha}  │  ${bm25Pct}%  │   ${vecPct}%  │ ${mrr}   │ ${bar}${medal}`);
    });

    // Best result detail
    const best = results[0];
    console.log('');
    console.log('═══════════════════════════════════════════════════════════════');
    console.log(`  🎯 OPTIMAL RATIO: α = ${best.alpha.toFixed(2)}`);
    console.log(`     BM25 Weight: ${((1 - best.alpha) * 100).toFixed(0)}%`);
    console.log(`     Vector Weight: ${(best.alpha * 100).toFixed(0)}%`);
    console.log(`     Average MRR: ${best.avgMrr.toFixed(4)}`);
    console.log('═══════════════════════════════════════════════════════════════');

    // Per-case breakdown for the winning alpha
    console.log('\n  📋 Per-Domain Results at Optimal Alpha:');
    console.log('  ──────────────────────────────────────────────────────');
    for (const detail of best.caseDetails) {
        const status = detail.rank > 0 ? `✅ rank #${detail.rank} (MRR=${detail.mrr.toFixed(2)})` : '❌ not in top 5';
        console.log(`     ${detail.domain.padEnd(22)} ${status}`);
    }
    console.log('');
}

main().catch(err => {
    console.error('\n❌ Fatal error:', err.message);
    console.error(err.stack);
    process.exit(1);
});
