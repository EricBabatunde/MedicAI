"use client";

import { useState } from "react";
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
} from "lucide-react";

export default function Dashboard() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [recentQueries] = useState([
    "Magnesium sulfate loading dose",
    "Severe eclampsia criteria",
    "Neonatal jaundice phototherapy thresholds",
    "Malaria in pregnancy treatment",
    "Postpartum haemorrhage management",
  ]);

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

          {/* Close button (mobile) */}
          <button
            onClick={() => setIsSidebarOpen(false)}
            className="ml-auto lg:hidden text-slate-400 hover:text-white transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* New Consultation */}
        <div className="px-4 py-4">
          <button className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-brand-500 to-brand-600 px-4 py-2.5 text-sm font-semibold text-white shadow-lg shadow-brand-500/20 hover:from-brand-400 hover:to-brand-500 transition-all duration-300">
            <PlusCircle className="h-4 w-4" />
            New Consultation
          </button>
        </div>

        {/* History */}
        <div className="flex-1 overflow-y-auto px-3 pb-3">
          <p className="px-2 pb-2 text-[11px] font-semibold uppercase tracking-widest text-slate-500">
            Recent
          </p>
          <ul className="space-y-1">
            {recentQueries.map((query, i) => (
              <li key={i}>
                <button className="glass-panel-hover flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-left text-sm text-slate-300">
                  <MessageSquare className="h-4 w-4 shrink-0 text-slate-500" />
                  <span className="truncate">{query}</span>
                </button>
              </li>
            ))}
          </ul>
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

        {/* Central search canvas */}
        <main className="flex-1 flex flex-col items-center justify-center px-6">
          <div className="w-full max-w-2xl text-center">
            <h1 className="text-3xl sm:text-4xl font-bold text-white tracking-tight mb-2">
              Clinical Triage Assistant
            </h1>
            <p className="text-slate-400 mb-8">
              Search across 19,034 medical knowledge chunks instantly.
            </p>

            {/* Search bar */}
            <div className="relative">
              <Search className="pointer-events-none absolute left-5 top-1/2 -translate-y-1/2 h-5 w-5 text-slate-500" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Describe symptoms or ask a clinical question…"
                className="glass-input w-full rounded-2xl py-4 pl-14 pr-6 text-lg placeholder:text-slate-600 focus:placeholder:text-slate-500"
              />
            </div>
          </div>
        </main>

        {/* Telemetry footer */}
        <footer className="border-t border-white/[0.06] px-5 py-3">
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