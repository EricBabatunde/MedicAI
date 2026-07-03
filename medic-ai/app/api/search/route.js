import { NextResponse } from 'next/server';
import { getEmbedding, cosineSimilarity } from '@/lib/embedding';
import medicalData from '@/data/medic_database.json';

// In-memory cache for the pre-computed database embeddings
let cachedDbEmbeddings = null;

export async function POST(req) {
    try {
        const { query } = await req.json();
        const records = medicalData.medical_records;

        // 1. Pre-compute DB embeddings (Runs only on the first request)
        if (!cachedDbEmbeddings) {
            console.log("Embedding database records into memory...");
            cachedDbEmbeddings = await Promise.all(records.map(async (record) => {
                // Combine relevant keys to create a rich searchable text string
                const searchableText = [
                    record.primary_topic,
                    ...(record.clinical_signs || []),
                    ...(record.indications || [])
                ].join(" ");

                const embedding = await getEmbedding(searchableText);
                return { ...record, embedding };
            }));
        }

        // 2. Embed the user's query
        const queryEmbedding = await getEmbedding(query);

        // 3. Calculate similarity and sort highest to lowest
        const scoredRecords = cachedDbEmbeddings.map(record => ({
            ...record,
            score: cosineSimilarity(queryEmbedding, record.embedding)
        })).sort((a, b) => b.score - a.score);

        // 4. Return the top 3 matches
        return NextResponse.json({ results: scoredRecords.slice(0, 3) });

    } catch (error) {
        return NextResponse.json({ error: error.message }, { status: 500 });
    }
}