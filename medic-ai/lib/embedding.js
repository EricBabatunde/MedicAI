import { pipeline } from '@huggingface/transformers';

class PipelineSingleton {
    static task = 'feature-extraction';
    static model = 'Xenova/all-MiniLM-L6-v2';
    static instance = null;

    static async getInstance(progress_callback = null) {
        if (this.instance === null) {
            this.instance = pipeline(this.task, this.model, { progress_callback });
        }
        return this.instance;
    }
}

export async function getEmbedding(text) {
    const extractor = await PipelineSingleton.getInstance();
    // Setting normalize: true means the dot product equals the cosine similarity
    const output = await extractor(text, { pooling: 'mean', normalize: true });
    return Array.from(output.data);
}

export function cosineSimilarity(vecA, vecB) {
    return vecA.reduce((sum, a, idx) => sum + a * vecB[idx], 0);
}