"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  FileText,
  AlertTriangle,
  Activity,
  CheckSquare,
  BookOpen,
  ChevronDown,
  ChevronUp,
  RotateCcw,
  Copy,
  Check,
  Sparkles,
} from "lucide-react";

function ScoreBadge({ score }) {
  const pct = Math.round((score ?? 0) * 100);
  let color = "text-slate-400 border-slate-500/30";
  if (pct >= 80) color = "text-emerald-400 border-emerald-500/30";
  else if (pct >= 50) color = "text-brand-300 border-brand-500/30";
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-semibold tabular-nums ${color}`}
    >
      {pct}% match
    </span>
  );
}

function CategoryTag({ category }) {
  if (!category) return null;
  const map = {
    diagnosis: "bg-violet-500/15 text-violet-300 border-violet-500/25",
    treatment: "bg-emerald-500/15 text-emerald-300 border-emerald-500/25",
    prevention: "bg-sky-500/15 text-sky-300 border-sky-500/25",
    procedure: "bg-amber-500/15 text-amber-300 border-amber-500/25",
  };
  const cls =
    map[category?.toLowerCase()] ||
    "bg-slate-500/15 text-slate-300 border-slate-500/25";
  return (
    <span
      className={`inline-block rounded-full border px-2.5 py-0.5 text-[10px] font-semibold capitalize ${cls}`}
    >
      {category}
    </span>
  );
}

function EvidenceCard({ result, index }) {
  const [expanded, setExpanded] = useState(false);
  const ctx = result.hierarchical_context || {};
  const title =
    result.source_text || ctx.chapter || ctx.primary_topic || "Clinical Source";
  const topic = [ctx.primary_topic, ctx.sub_topic].filter(Boolean).join(" → ");
  const pages = result.page_reference?.length
    ? `pp. ${result.page_reference.join(", ")}`
    : null;

  return (
    <button
      type="button"
      onClick={() => setExpanded(!expanded)}
      className="glass-panel glass-panel-hover w-full rounded-xl p-4 cursor-pointer text-left"
    >
      {/* Collapsed header */}
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 flex-wrap mb-1">
            <span className="text-xs font-bold text-brand-400">
              #{index + 1}
            </span>
            <CategoryTag category={result.clinical_category} />
            <ScoreBadge score={result.scores?.hybrid} />
          </div>
          <p className="text-sm font-semibold text-slate-100 truncate">
            {title}
          </p>
          {topic && (
            <p className="text-xs text-slate-400 truncate mt-0.5">{topic}</p>
          )}
          {pages && (
            <p className="text-[11px] text-slate-500 mt-0.5">{pages}</p>
          )}
        </div>
        {expanded ? (
          <ChevronUp className="h-4 w-4 shrink-0 text-slate-500" />
        ) : (
          <ChevronDown className="h-4 w-4 shrink-0 text-slate-500" />
        )}
      </div>

      {/* Expanded excerpt */}
      {expanded && (
        <div className="mt-3 bg-black/40 p-4 rounded-lg font-mono text-xs text-slate-300 whitespace-pre-wrap leading-relaxed max-h-64 overflow-y-auto">
          {result.text_content}
        </div>
      )}
    </button>
  );
}

export default function ClinicalReport({ data, onReset, query }) {
  const [copied, setCopied] = useState(false);

  const domain = data.routed_domain?.replace(/_/g, " ") || "General Medicine";
  const pipelineMs = data.pipeline_ms;

  function handleCopy() {
    const text = [
      `Query: ${query}`,
      `Domain: ${domain}`,
      "",
      "── Clinical Assessment ──",
      data.clinical_assessment,
      "",
      "── Treatment Plan ──",
      data.treatment_plan,
      "",
      data.critical_warnings ? "── Critical Warnings ──" : "",
      data.critical_warnings || "",
    ]
      .filter(Boolean)
      .join("\n");

    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }

  return (
    <div className="w-full max-w-5xl mx-auto space-y-6 pb-12">
      {/* ── Header & Meta Ribbon ── */}
      <div className="glass-panel rounded-2xl p-6">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          {/* Query & domain */}
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2 mb-2">
              <Sparkles className="h-5 w-5 text-brand-400 shrink-0" />
              <h2 className="text-lg font-bold text-white truncate">{query}</h2>
            </div>
            <div className="flex items-center gap-3 flex-wrap">
              <span className="bg-brand-500/10 text-brand-300 border border-brand-500/20 px-3 py-1 rounded-full text-xs font-semibold capitalize">
                Domain: {domain}
              </span>
              {pipelineMs && (
                <span className="text-[11px] text-slate-500">
                  {(pipelineMs / 1000).toFixed(1)}s pipeline
                </span>
              )}
            </div>
          </div>

          {/* Action bar */}
          <div className="flex items-center gap-2 shrink-0">
            <button
              onClick={handleCopy}
              className="glass-panel glass-panel-hover flex items-center gap-1.5 rounded-lg px-3 py-2 text-xs font-medium text-slate-300"
            >
              {copied ? (
                <Check className="h-3.5 w-3.5 text-emerald-400" />
              ) : (
                <Copy className="h-3.5 w-3.5" />
              )}
              {copied ? "Copied" : "Copy Report"}
            </button>
            <button
              onClick={onReset}
              className="flex items-center gap-1.5 rounded-lg bg-gradient-to-r from-brand-500 to-brand-600 px-4 py-2 text-xs font-semibold text-white shadow-lg shadow-brand-500/20 hover:from-brand-400 hover:to-brand-500 transition-all duration-300"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              New Consultation
            </button>
          </div>
        </div>
      </div>

      {/* ── Section 1: Critical Warnings ── */}
      {data.critical_warnings &&
        data.critical_warnings.trim() &&
        !data.critical_warnings.toLowerCase().includes("no critical") &&
        !data.critical_warnings.toLowerCase().includes("no specific critical") && (
          <div className="bg-rose-500/10 border border-rose-500/30 text-rose-100 rounded-2xl p-5 shadow-[0_0_20px_rgba(244,63,94,0.15)]">
            <div className="flex items-start gap-3">
              <div className="relative shrink-0 mt-0.5">
                <AlertTriangle className="h-5 w-5 text-rose-400" />
                <span className="absolute -top-0.5 -right-0.5 h-2 w-2 rounded-full bg-rose-400 animate-ping" />
              </div>
              <div className="min-w-0 flex-1">
                <h3 className="text-sm font-bold text-rose-200 mb-2">
                  Critical Warnings
                </h3>
                <div className="prose prose-invert prose-sm max-w-none text-rose-100/90 leading-relaxed">
                  <ReactMarkdown>{data.critical_warnings}</ReactMarkdown>
                </div>
              </div>
            </div>
          </div>
        )}

      {/* ── Section 2: Clinical Assessment ── */}
      {data.clinical_assessment && (
        <div className="glass-panel rounded-2xl p-6 border-l-2 border-brand-500/40">
          <div className="flex items-center gap-2 mb-4">
            <Activity className="h-5 w-5 text-brand-400" />
            <h3 className="text-base font-bold text-white">
              Clinical Assessment
            </h3>
          </div>
          <div className="prose prose-invert max-w-none text-slate-100 leading-relaxed">
            <ReactMarkdown>{data.clinical_assessment}</ReactMarkdown>
          </div>
        </div>
      )}

      {/* ── Section 3: Treatment Plan ── */}
      {data.treatment_plan && (
        <div className="glass-panel rounded-2xl p-6">
          <div className="flex items-center gap-2 mb-4">
            <CheckSquare className="h-5 w-5 text-emerald-400" />
            <h3 className="text-base font-bold text-white">
              Recommended Clinical Action Plan
            </h3>
          </div>
          <div className="prose prose-invert max-w-none text-slate-200 leading-relaxed [&_li]:marker:text-brand-400">
            <ReactMarkdown>{data.treatment_plan}</ReactMarkdown>
          </div>
        </div>
      )}

      {/* ── Section 4: Evidence & Citations ── */}
      {data.results && data.results.length > 0 && (
        <div>
          <div className="flex items-center gap-2 mb-4 px-1">
            <BookOpen className="h-5 w-5 text-brand-400" />
            <h3 className="text-base font-bold text-white">
              Retrieved Clinical Evidence &amp; Guidelines
            </h3>
            <span className="text-xs text-slate-500 ml-1">
              ({data.results.length} sources)
            </span>
          </div>
          <div className="space-y-3">
            {data.results.map((result, i) => (
              <EvidenceCard key={result.chunk_id || i} result={result} index={i} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

