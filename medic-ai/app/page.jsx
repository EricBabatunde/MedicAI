export default function Dashboard() {
  return (
    <main className="h-full w-full flex items-center justify-center p-8">
      <div className="glass-panel rounded-2xl p-12 max-w-lg text-center flex flex-col items-center gap-4">
        <h1 className="text-2xl font-bold text-white tracking-tight">
          MedicAI Theme Active
        </h1>
        <p className="text-slate-400">
          The deep slate mesh gradient and frosted glass utilities are working
          perfectly.
        </p>
        <input
          type="text"
          placeholder="Test input..."
          className="glass-input rounded-full px-6 py-3 w-full mt-4"
        />
      </div>
    </main>
  );
}