import { Plus_Jakarta_Sans } from "next/font/google";
import "./globals.css";

const plusJakarta = Plus_Jakarta_Sans({
  subsets: ["latin"],
  variable: "--font-plus-jakarta",
  weight: ["300", "400", "500", "600", "700", "800"],
  display: "swap",
});

export const metadata = {
  title: "MedicAI - Clinical Triage Assistant",
  description: "Offline-first medical RAG assistant",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={`${plusJakarta.variable} h-screen antialiased`}>
      <body className="overflow-hidden h-screen bg-[#0f172a] text-slate-100 antialiased font-sans">
        {children}
      </body>
    </html>
  );
}
