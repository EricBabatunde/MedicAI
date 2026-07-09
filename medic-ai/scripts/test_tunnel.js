#!/usr/bin/env node
/**
 * ═══════════════════════════════════════════════════════════════════════
 *  Tunnel Diagnostic Script — Probes Cloudflare / llama.cpp endpoints
 *  Usage:  node scripts/test_tunnel.js
 * ═══════════════════════════════════════════════════════════════════════
 */

// ── Paste your raw Cloudflare tunnel URL here (NO trailing slash) ────
const BASE_URL = 'https://tommy-portraits-inclusive-viewed.trycloudflare.com';

async function main() {
    console.log('═══════════════════════════════════════════════════════════════');
    console.log('  🔍 TUNNEL DIAGNOSTIC — Probing LLM server endpoints');
    console.log(`  🌐 Base URL: ${BASE_URL}`);
    console.log('═══════════════════════════════════════════════════════════════\n');

    // ── Test 1: Health Check (/health) ──────────────────────────────
    console.log('── TEST 1: Health Check (/health) ────────────────────────────');
    try {
        const res = await fetch(`${BASE_URL}/health`);
        const body = await res.text();
        console.log(`   Status: ${res.status} ${res.statusText}`);
        console.log(`   Body:   ${body}`);
        console.log(`   ${res.ok ? '✅ PASSED' : '❌ FAILED'}\n`);
    } catch (err) {
        console.log(`   ❌ NETWORK ERROR: ${err.message}\n`);
    }

    // ── Test 2: Models Check (/v1/models) ───────────────────────────
    console.log('── TEST 2: Models Check (/v1/models) ─────────────────────────');
    try {
        const res = await fetch(`${BASE_URL}/v1/models`);
        const body = await res.text();
        console.log(`   Status: ${res.status} ${res.statusText}`);
        console.log(`   Body:   ${body}`);
        console.log(`   ${res.ok ? '✅ PASSED' : '❌ FAILED'}\n`);
    } catch (err) {
        console.log(`   ❌ NETWORK ERROR: ${err.message}\n`);
    }

    // ── Test 3: Chat Endpoint (/v1/chat/completions) ────────────────
    console.log('── TEST 3: Chat Completions (/v1/chat/completions) ───────────');
    try {
        const res = await fetch(`${BASE_URL}/v1/chat/completions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                messages: [{ role: 'user', content: 'Test ping' }],
                max_tokens: 10,
            }),
        });
        const body = await res.text();
        console.log(`   Status: ${res.status} ${res.statusText}`);
        console.log(`   Body:   ${body}`);
        console.log(`   ${res.ok ? '✅ PASSED' : '❌ FAILED'}\n`);
    } catch (err) {
        console.log(`   ❌ NETWORK ERROR: ${err.message}\n`);
    }

    console.log('═══════════════════════════════════════════════════════════════');
    console.log('  🏁 Diagnostic complete');
    console.log('═══════════════════════════════════════════════════════════════');
}

main();
