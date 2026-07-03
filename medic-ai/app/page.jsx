'use client';

import { useState } from 'react';

export default function DiagnosticTestBench() {
    const [query, setQuery] = useState('');
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState([]);
    const [error, setError] = useState('');

    const handleSearch = async (e) => {
        e.preventDefault();
        if (!query.trim()) return;

        setLoading(true);
        setError('');
        try {
            const response = await fetch('/api/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query }),
            });

            const data = await response.json();
            if (data.error) throw new Error(data.error);

            setResults(data.results || []);
        } catch (err) {
            setError(err.message || 'Failed to fetch results');
        } finally {
            setLoading(false);
        }
    };

    // Helper to color-code the triage badges
    const getBadgeStyle = (level) => {
        const lvl = level?.toLowerCase() || '';
        if (lvl.includes('red') || lvl.includes('emergency')) {
            return 'bg-red-100 text-red-800 border-red-200';
        }
        if (lvl.includes('yellow') || lvl.includes('priority')) {
            return 'bg-yellow-100 text-yellow-800 border-yellow-200';
        }
        return 'bg-green-100 text-green-800 border-green-200';
    };

    return (
        <main className="min-h-screen bg-slate-50 py-10 px-4 sm:px-6 lg:px-8 font-sans">
            <div className="max-w-4xl mx-auto">

                {/* Header */}
                <div className="text-center mb-10">
                    <h1 className="text-3xl font-bold text-slate-900 tracking-tight">
                        Clinical Knowledge Base Test Bench
                    </h1>
                    <p className="mt-2 text-sm text-slate-600">
                        Phase 1: Validating Semantic Retrieval & Database Intent Matching
                    </p>
                </div>

                {/* Input Form */}
                <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 mb-8">
                    <form onSubmit={handleSearch} className="space-y-4">
                        <div>
                            <label htmlFor="symptoms" className="block text-sm font-medium text-slate-700 mb-2">
                                Enter Patient Symptoms or Clinical Scenario
                            </label>
                            <textarea
                                id="symptoms"
                                rows={3}
                                className="w-full rounded-lg border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 text-slate-900 p-3 border"
                                placeholder="e.g., patient is a pregnant woman presenting with a severe headache and high blood pressure..."
                                value={query}
                                onChange={(e) => setQuery(e.target.value)}
                            />
                        </div>
                        <button
                            type="submit"
                            disabled={loading}
                            className="w-full flex justify-center py-2.5 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-indigo-400 transition-colors"
                        >
                            {loading ? 'Computing Math Vectors & Searching...' : 'Test Database Match'}
                        </button>
                    </form>

                    {error && (
                        <div className="mt-4 p-3 bg-red-50 text-red-700 text-sm rounded-lg border border-red-200">
                            {error}
                        </div>
                    )}
                </div>

                {/* Results Matrix */}
                <div className="space-y-6">
                    <h2 className="text-lg font-semibold text-slate-900">
                        Top Database Returns ({results.length})
                    </h2>

                    {results.length === 0 && !loading && (
                        <p className="text-slate-500 text-center py-8 bg-white rounded-xl border border-dashed border-slate-300">
                            No tests executed yet. Enter symptoms above to evaluate vector alignment scores.
                        </p>
                    )}

                    {results.map((record, index) => (
                        <div
                            key={record.id}
                            className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden transition-all hover:border-slate-300"
                        >
                            {/* Card Title Bar */}
                            <div className="bg-slate-50 border-b border-slate-200 px-6 py-4 flex flex-wrap items-center justify-between gap-2">
                                <div className="flex items-center gap-3">
                                    <span className="text-xs font-mono font-bold px-2 py-1 bg-slate-200 text-slate-700 rounded">
                                        Rank #{index + 1}
                                    </span>
                                    <h3 className="text-base font-bold text-slate-900">{record.primary_topic}</h3>
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className="text-xs text-slate-500 font-mono">Score: {(record.score * 100).toFixed(1)}%</span>
                                    {record.triage_level && (
                                        <span className={`text-xs font-bold uppercase px-2.5 py-1 rounded-full border ${getBadgeStyle(record.triage_level)}`}>
                                            {record.triage_level}
                                        </span>
                                    )}
                                </div>
                            </div>

                            {/* Card Body */}
                            <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">

                                {/* Left Column: Criteria */}
                                <div className="space-y-4">
                                    <div>
                                        <h4 className="font-semibold text-slate-700 mb-1">Database ID / Type</h4>
                                        <p className="text-slate-600 font-mono text-xs">{record.id} / <span className="underline">{record.record_type}</span></p>
                                    </div>
                                    {record.clinical_signs && record.clinical_signs.length > 0 && (
                                        <div>
                                            <h4 className="font-semibold text-slate-700 mb-1.5">Target Clinical Signs</h4>
                                            <ul className="list-disc pl-5 space-y-1 text-slate-600">
                                                {record.clinical_signs.map((sign, i) => <li key={i}>{sign}</li>)}
                                            </ul>
                                        </div>
                                    )}
                                </div>

                                {/* Right Column: Dynamic Content Based on Type */}
                                <div className="space-y-4 border-t md:border-t-0 md:border-l border-slate-100 md:pl-6">
                                    {record.immediate_actions && record.immediate_actions.length > 0 && (
                                        <div>
                                            <h4 className="font-semibold text-slate-700 mb-1.5">Immediate Directives</h4>
                                            <ul className="list-decimal pl-5 space-y-1 text-slate-600 font-medium">
                                                {record.immediate_actions.map((action, i) => <li key={i}>{action}</li>)}
                                            </ul>
                                        </div>
                                    )}
                                    {record.adult_dosing && (
                                        <div>
                                            <h4 className="font-semibold text-slate-700 mb-1">Adult Formulation & Dosing</h4>
                                            <p className="text-slate-600 bg-indigo-50/50 p-2.5 rounded-lg border border-indigo-100/50">{record.adult_dosing}</p>
                                        </div>
                                    )}
                                    {record.step_by_step_guide && record.step_by_step_guide.length > 0 && (
                                        <div>
                                            <h4 className="font-semibold text-slate-700 mb-1.5">Execution Guide</h4>
                                            <ol className="list-decimal pl-5 space-y-1 text-slate-600">
                                                {record.step_by_step_guide.map((step, i) => <li key={i}>{step}</li>)}
                                            </ol>
                                        </div>
                                    )}
                                    {record.referral_note && (
                                        <div className="pt-2">
                                            <h4 className="font-semibold text-slate-700 mb-1">Clinical Protocol Notes</h4>
                                            <p className="text-xs text-slate-500 italic bg-slate-50 p-2 rounded border border-slate-100">{record.referral_note}</p>
                                        </div>
                                    )}
                                </div>

                            </div>
                        </div>
                    ))}
                </div>

            </div>
        </main>
    );
}