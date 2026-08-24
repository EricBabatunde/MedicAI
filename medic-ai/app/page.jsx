"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import {
  Activity,
  PlusCircle,
  MessageSquare,
  Search,
  Menu,
  X,
  Server,
  Database,
  Zap,
  Loader2,
  Radio,
} from "lucide-react";
import ClinicalReport from "@/components/ClinicalReport";

export default function Dashboard() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [recentQueries, setRecentQueries] = useState([]);

  const [isLoading, setIsLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState("");
  const [clinicalResult, setClinicalResult] = useState(null);
  const [activeQuery, setActiveQuery] = useState("");
  const [error, setError] = useState(null);

  // SSE streaming state
  const [isStreamingTokens, setIsStreamingTokens] = useState(false);
  const [streamedText, setStreamedText] = useState("");
  const [retrievedEvidence, setRetrievedEvidence] = useState([]);

  const streamTextRef = useRef("");
  const evidenceRef = useRef([]);

  // ── Load history from localStorage on mount ──
  useEffect(() => {
    try {
      const stored = localStorage.getItem("medicai_history");
      if (stored) {
        const parsed = JSON.parse(stored);
        if (Array.isArray(parsed)) setRecentQueries(parsed);
      }
    } catch {
      /* ignore corrupt data */
    }
  }, []);

  // ── Search handler (SSE consumer) ──
  const handleSearch = useCallback(
    async (e, overrideQuery) => {
      if (e) e.preventDefault();
      const trimmed = (overrideQuery ?? searchQuery).trim();
      if (!trimmed) return;

      // Reset all state
      setIsLoading(true);
      setError(null);
      setClinicalResult(null);
      setActiveQuery(trimmed);
      setIsSidebarOpen(false);
      setIsStreamingTokens(false);
      setStreamedText("");
      setRetrievedEvidence([]);
      setLoadingStep("Connecting to pipeline...");
      streamTextRef.current = "";
      evidenceRef.current = [];

      try {
        const response = await fetch("/api/search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: trimmed }),
        });

        if (!response.ok) {
          throw new Error(`Server responded with ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let completeMeta = {};

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            const trimmedLine = line.trim();
            if (!trimmedLine.startsWith("data: ")) continue;

            let data;
            try {
              data = JSON.parse(trimmedLine.slice(6));
            } catch {
              continue;
            }

            if (data.type === "status") {
              setLoadingStep(data.step);
            } else if (data.type === "token") {
              setIsStreamingTokens(true);
              streamTextRef.current += data.text;
              setStreamedText(streamTextRef.current);
            } else if (data.type === "complete") {
              evidenceRef.current = data.results || [];
              setRetrievedEvidence(evidenceRef.current);
              completeMeta = data;
            } else if (data.type === "error") {
              throw new Error(data.message || "Pipeline error");
            }
          }
        }

        // Stream finished — parse accumulated LLM text into structured result
        const rawLLM = streamTextRef.current;
        let parsedClinical;

        try {
          const cleaned = rawLLM
            .replace(/```json\s*/gi, "")
            .replace(/```\s*/g, "")
            .trim();
          parsedClinical = JSON.parse(cleaned);
        } catch {
          console.warn("JSON parse failed on streamed text, using raw");
          parsedClinical = {
            clinical_assessment: rawLLM,
            treatment_plan: "Unable to parse structured response.",
            critical_warnings: "",
            references: [],
          };
        }

        // Merge with metadata from 'complete' event
        setClinicalResult({
          ...parsedClinical,
          routed_domain: completeMeta.routed_domain,
          domain_chunk_count: completeMeta.domain_chunk_count,
          total_chunk_count: completeMeta.total_chunk_count,
          bm25_candidates: completeMeta.bm25_candidates,
          pipeline_ms: completeMeta.pipeline_ms,
          results: evidenceRef.current,
        });

        // Persist to history
        setRecentQueries((prev) => {
          const updated = [
            trimmed,
            ...prev.filter((q) => q !== trimmed),
          ].slice(0, 20);
          localStorage.setItem("medicai_history", JSON.stringify(updated));
          return updated;
        });

        setSearchQuery("");
      } catch (err) {
        setError(err.message || "Something went wrong");
      } finally {
        setIsLoading(false);
        setIsStreamingTokens(false);
        setLoadingStep("");
      }
    },
    [searchQuery]
  );

  function handleReset() {
    setClinicalResult(null);
    setActiveQuery("");
    setError(null);
    setSearchQuery("");
    setStreamedText("");
    setRetrievedEvidence([]);
    setIsStreamingTokens(false);
  }

  // ── Decide what the central area shows ──
  const showSearch = !isLoading && !clinicalResult;
  const showLoading = isLoading && !isStreamingTokens;
  const showStreaming = isLoading && isStreamingTokens;
  const showResult = !isLoading && clinicalResult;

  return (
    <div className="flex h-screen w-full relative">
      {/* ── Mobile overlay ── */}
      {isSidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/80 backdrop-blur-sm lg:hidden"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* ── Sidebar ── */}
      <aside
        className={`
          glass-panel fixed inset-y-0 left-0 z-40 w-64 flex flex-col
          transform transition-transform duration-300 ease-in-out
          lg:relative lg:translate-x-0
          ${isSidebarOpen ? "translate-x-0" : "-translate-x-full"}
        `}
      >
        {/* Sidebar header */}
        <div className="flex items-center gap-3 px-5 py-5 border-b border-white/[0.06]">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-brand-400 to-brand-600">
            <Activity className="h-5 w-5 text-white" />
          </div>
          <span className="text-lg font-bold text-white tracking-tight">
            MedicAI
          </span>
          <button
            onClick={() => setIsSidebarOpen(false)}
            className="ml-auto lg:hidden text-slate-400 hover:text-white transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* New Consultation */}
        <div className="px-4 py-4">
          <button
            onClick={handleReset}
            className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-brand-500 to-brand-600 px-4 py-2.5 text-sm font-semibold text-white shadow-lg shadow-brand-500/20 hover:from-brand-400 hover:to-brand-500 transition-all duration-300"
          >
            <PlusCircle className="h-4 w-4" />
            New Consultation
          </button>
        </div>

        {/* History */}
        <div className="flex-1 overflow-y-auto px-3 pb-3">
          <p className="px-2 pb-2 text-[11px] font-semibold uppercase tracking-widest text-slate-500">
            Recent
          </p>
          {recentQueries.length === 0 ? (
            <p className="px-2 text-xs text-slate-600 italic">
              No history yet
            </p>
          ) : (
            <ul className="space-y-1">
              {recentQueries.map((query, i) => (
                <li key={i}>
                  <button
                    onClick={() => handleSearch(null, query)}
                    className="glass-panel-hover flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-left text-sm text-slate-300"
                  >
                    <MessageSquare className="h-4 w-4 shrink-0 text-slate-500" />
                    <span className="truncate">{query}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* User profile */}
        <div className="border-t border-white/[0.06] px-4 py-4">
          <div className="glass-panel glass-panel-hover flex items-center gap-3 rounded-xl px-3 py-2.5 cursor-pointer">
            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-brand-400 to-brand-600 text-xs font-bold text-white">
              BE
            </div>
            <div className="min-w-0">
              <p className="truncate text-sm font-medium text-white">
                Babatunde Eric
              </p>
              <p className="truncate text-xs text-slate-500">System Admin</p>
            </div>
          </div>
        </div>
      </aside>

      {/* ── Main content ── */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top header */}
        <header className="flex items-center justify-between px-5 py-3 border-b border-white/[0.06]">
          <button
            onClick={() => setIsSidebarOpen(true)}
            className="lg:hidden text-slate-400 hover:text-white transition-colors"
          >
            <Menu className="h-5 w-5" />
          </button>

          <div className="ml-auto flex items-center gap-2 text-xs text-slate-400">
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-green-400 opacity-75" />
              <span className="relative inline-flex h-2 w-2 rounded-full bg-green-500" />
            </span>
            System Online
          </div>
        </header>

        {/* Central area */}
        <main className="flex-1 flex flex-col items-center overflow-y-auto px-6 py-8">
          {/* ── Search canvas ── */}
          {showSearch && (
            <div className="flex-1 flex flex-col items-center justify-center w-full max-w-2xl text-center">
              <h1 className="text-3xl sm:text-4xl font-bold text-white tracking-tight mb-2">
                Clinical Triage Assistant
              </h1>
              <p className="text-slate-400 mb-8">
                Search across 19,034 medical knowledge chunks instantly.
              </p>

              <form onSubmit={handleSearch} className="relative w-full">
                <Search className="pointer-events-none absolute left-5 top-1/2 -translate-y-1/2 h-5 w-5 text-slate-500" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Describe symptoms or ask a clinical question…"
                  className="glass-input w-full rounded-2xl py-4 pl-14 pr-6 text-lg placeholder:text-slate-600 focus:placeholder:text-slate-500"
                />
              </form>

              {error && (
                <p className="mt-4 text-sm text-red-400">⚠ {error}</p>
              )}
            </div>
          )}

          {/* ── Pipeline status spinner (pre-token phase) ── */}
          {showLoading && (
            <div className="flex-1 flex items-center justify-center w-full">
              <div className="glass-panel p-8 rounded-2xl max-w-md w-full flex flex-col items-center gap-6">
                <Loader2 className="w-12 h-12 text-brand-400 animate-spin" />
                <p className="text-brand-300 font-medium text-lg text-center">
                  {loadingStep}
                </p>
                <div className="w-full h-1 bg-white/10 rounded-full overflow-hidden">
                  <div className="bg-brand-500 h-full w-1/2 animate-pulse rounded-full" />
                </div>
              </div>
            </div>
          )}

          {/* ── Live Synthesis Terminal (streaming tokens) ── */}
          {showStreaming && (
            <div className="flex-1 flex items-center justify-center w-full">
              <div className="glass-panel p-8 rounded-2xl max-w-2xl w-full flex flex-col gap-5">
                {/* Header */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <Radio className="h-5 w-5 text-emerald-400" />
                    <h3 className="text-base font-bold text-white">
                      Live Synthesis
                    </h3>
                  </div>
                  <div className="flex items-center gap-2 text-xs text-emerald-400">
                    <span className="relative flex h-2 w-2">
                      <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
                      <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-500" />
                    </span>
                    GPU Connection Active
                  </div>
                </div>

                {/* Streaming output */}
                <pre className="font-mono text-sm text-brand-300 whitespace-pre-wrap overflow-y-auto max-h-96 bg-black/30 rounded-xl p-4 leading-relaxed">
                  {streamedText}
                  <span className="inline-block w-2 h-4 bg-brand-400 animate-pulse ml-0.5 align-middle" />
                </pre>

                {/* Bottom bar */}
                <div className="flex items-center justify-between text-xs text-slate-500">
                  <span>{streamedText.length} characters received</span>
                  <span className="text-brand-400 font-medium">
                    {loadingStep}
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* ── Clinical report ── */}
          {showResult && (
            <ClinicalReport
              data={clinicalResult}
              onReset={handleReset}
              query={activeQuery}
            />
          )}
        </main>

        {/* Telemetry footer */}
        <footer className="border-t border-white/[0.06] px-5 py-3 shrink-0">
          <div className="flex flex-wrap items-center justify-center gap-x-8 gap-y-2 text-xs text-slate-500">
            <span className="flex items-center gap-1.5">
              <Database className="h-3.5 w-3.5" />
              19,034 Local Chunks
            </span>
            <span className="flex items-center gap-1.5">
              <Server className="h-3.5 w-3.5" />
              Phi-4-mini Active
            </span>
            <span className="flex items-center gap-1.5">
              <Zap className="h-3.5 w-3.5" />
              Africa Deep Tech Challenge 2026
            </span>
          </div>
        </footer>
      </div>
    </div>
  );
}