import React, { useState, useRef } from 'react';

// --- THE API BRIDGE ---
const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000/api";

const callBackend = async (endpoint, body, isFile = false) => {
  const options = {
    method: 'POST',
    body: isFile ? body : JSON.stringify(body),
  };
  if (!isFile) options.headers = { 'Content-Type': 'application/json' };

  const res = await fetch(`${API_BASE}${endpoint}`, options);
  if (!res.ok) {
    const errorText = await res.text();
    let detail = "Backend connection failed";
    try {
      const json = JSON.parse(errorText);
      if (json.detail) detail = json.detail;
    } catch (e) {
      detail = errorText || detail; // Use text if not JSON
    }
    throw new Error(detail);
  }
  return res.json();
};

// --- HELPER: SUM INSURED ANALYSIS ---
const SI_ANALYSIS_DATA = [
  {
    max: 10,
    status: "❌ Very Low Coverage",
    range_desc: "Your coverage is under ₹10 Lakhs.",
    reality: (policy) => {
      const isFamily = policy.policy_holders && policy.policy_holders.length > 1;
      const who = isFamily ? "your family" : "an individual";
      return [
        `The current sum insured is significantly below recommended market standards for ${who}. A single critical hospitalization (e.g., Cardiac or Neuro) often exceeds ₹8 Lakhs, resulting in potential out-of-pocket expenses.`,
        "With medical inflation at ~15% annually, the purchasing power of this coverage diminishes rapidly, potentially leaving you underinsured within 3 years.",
        "Advisory: This level of coverage exposes your financial portfolio to significant risk in the event of a major medical emergency."
      ];
    },
    color: "rose"
  },
  {
    max: 25,
    status: "⚠️ Basic to Moderate Coverage",
    range_desc: "Your coverage is between ₹10 Lakhs - ₹25 Lakhs.",
    reality: (policy) => {
      const isFamily = policy.policy_holders && policy.policy_holders.length > 1;
      const familySize = policy.policy_holders?.length || 1;

      if (isFamily) {
        return [
          `While this provides a basic safety net, it may be insufficient for a family of ${familySize}. Concurrent hospitalizations could exhaust the entire sum insured.`,
          "Coverage may be limited for advanced medical procedures such as Immunotherapy (₹20L+) or Robotic Surgeries, which are becoming standard care.",
          "For premium hospital admissions, the room rent eligibility associated with this sum insured may impose restrictions, leading to proportionate deductions."
        ];
      } else {
        return [
          "Provides a foundational safety net for an individual but may not fully cover high-cost critical illnesses or prolonged hospitalization.",
          "Coverage may be limited for advanced medical procedures such as Immunotherapy (₹20L+) or Robotic Surgeries, which are becoming standard care.",
          "For premium hospital admissions, the room rent eligibility associated with this sum insured may impose restrictions, leading to proportionate deductions."
        ];
      }
    },
    color: "amber"
  },
  {
    max: 50,
    status: "✅ Ideal & Recommended Coverage",
    range_desc: "Your coverage is between ₹25 Lakhs - ₹50 Lakhs.",
    reality: (policy) => {
      const isFamily = policy.policy_holders && policy.policy_holders.length > 1;
      const who = isFamily ? "your family" : "you";
      return [
        `This tier offers comprehensive financial protection for ${who}, ensuring access to approximately 95% of network hospitals without significant deductions.`,
        "The coverage is well-calibrated to handle high-cost procedures, including Organ Transplants and Cardiac surgeries.",
        "Your financial liability is minimized. Recommendation: Focus on optimizing 'Wellness Benefits' and 'OPD coverage' to enhance this portfolio."
      ];
    },
    color: "emerald"
  },
  {
    max: 9999, // > 50L
    status: "⭐ Optimal / Future-Ready Coverage",
    range_desc: "Your coverage is above ₹50 Lakhs.",
    reality: (policy) => {
      const isFamily = policy.policy_holders && policy.policy_holders.length > 1;
      const subject = isFamily ? "your family" : "you";
      return [
        `This is a premier tier of coverage, effectively future-proofing ${subject} against medical inflation for the next decade.`,
        "Ensures access to advanced global treatments (subject to policy terms) and premium hospital accommodations.",
        "Provides robust financial security, ensuring that long-term critical care or multiple claims do not impact your wealth preservation goals."
      ];
    },
    color: "purple"
  }
];

const getSumInsuredAnalysis = (amountStr) => {
  if (!amountStr) return null;
  // Parse amount string (e.g. "5L", "500000", "1 Cr", "1.5 Crore") into Lakhs
  let lakhs = 0;
  const lower = amountStr.toString().toLowerCase().replace(/,/g, '').trim();

  try {
    if (lower.includes('cr')) {
      const num = parseFloat(lower.replace(/[^0-9.]/g, ''));
      lakhs = num * 100;
    } else if (lower.includes('l') || lower.includes('lakh')) {
      const num = parseFloat(lower.replace(/[^0-9.]/g, ''));
      lakhs = num; // It's already in lakhs
    } else {
      // Assume raw number
      const num = parseFloat(lower.replace(/[^0-9.]/g, ''));
      if (num < 1000) lakhs = num; // Assume lakhs if small number? No, risky. 
      // Safest to assume raw rupees if > 1000
      else lakhs = num / 100000;
    }
  } catch (e) { return null; }

  // Find range
  return SI_ANALYSIS_DATA.find(d => lakhs < d.max) || SI_ANALYSIS_DATA[SI_ANALYSIS_DATA.length - 1];
};

// --- THE MAIN APP ---
export default function App() {
  const [policy, setPolicy] = useState({ company: '', premium: '' });
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState({ extracting: false, comparing: false });
  const [comparingItem, setComparingItem] = useState(null); // [NEW] State for comparison modal
  const fileRef = useRef();
  const extractedDetailsRef = useRef(null);
  const reportSectionRef = useRef(null);

  // Auto-scroll logic
  React.useEffect(() => {
    if (policy.company && extractedDetailsRef.current) {
      setTimeout(() => extractedDetailsRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' }), 100);
    }
  }, [policy]);

  React.useEffect(() => {
    if (report && reportSectionRef.current) {
      setTimeout(() => reportSectionRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
    }
  }, [report]);

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setLoading({ ...loading, extracting: true });
    try {
      const fd = new FormData();
      fd.append('file', file);
      const data = await callBackend("/extract", fd, true);
      setPolicy(data);
    } catch (err) { alert(err.message); }
    finally { setLoading({ ...loading, extracting: false }); }
  };

  const handleCompare = async () => {
    setLoading({ ...loading, comparing: true });
    try {
      const data = await callBackend("/compare", policy);
      setReport(data);
    } catch (err) { alert(err.message); }
    finally { setLoading({ ...loading, comparing: false }); }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 text-slate-900 font-sans">
      <header className="px-10 py-5 bg-white/80 backdrop-blur-md border-b border-indigo-100 font-bold text-xl flex items-center justify-between print:hidden sticky top-0 z-50">
        <div className="flex items-center gap-4">
          <img src="/logo3.png" alt="Share India" className="h-9 object-contain" />
        </div>
        <div className="flex items-center gap-6 text-sm font-medium text-slate-500">
          <div className="flex items-center gap-2">
            <span className="bg-slate-100 p-2 rounded-full">📞</span>
            <div>
              <p className="text-xs text-slate-400 uppercase tracking-wider">Support</p>
              <p className="text-slate-800">+91-99999-00000</p>
            </div>
          </div>
          <div className="h-8 w-px bg-slate-200"></div>
          <a href="#" className="hover:text-blue-600 transition">Help Center</a>
        </div>
      </header>

      <main className={`max-w-6xl mx-auto py-12 px-6 ${comparingItem ? 'print:hidden' : ''}`}>

        {/* [NEW] Centered Heading */}
        <div className="text-center mb-10">
          <h1 className="text-4xl font-black bg-clip-text text-transparent bg-gradient-to-r from-blue-500 to-indigo-500 mb-2 pb-1 leading-normal">PolicyWise</h1>
          <p className="text-slate-500 font-medium">Smart AI Insurance Analysis</p>
        </div>

        {/* 2. Upload Box - Redesigned */}
        <div className="bg-white rounded-3xl p-10 shadow-xl border border-slate-100 max-w-3xl mx-auto print:hidden transition-all hover:shadow-2xl">

          <div className="text-center mb-8">
            <h2 className="text-3xl font-black text-slate-800 mb-2">Upload Your Policy</h2>
            <p className="text-slate-500 text-lg">AI-powered analysis & comparison in seconds.</p>
          </div>

          <div
            onClick={() => fileRef.current.click()}
            className="border-2 border-dashed border-blue-200 bg-blue-50/30 rounded-2xl p-16 text-center cursor-pointer hover:bg-blue-50 hover:border-blue-400 transition group"
          >
            <input type="file" ref={fileRef} onChange={handleUpload} className="hidden" />
            <div className="text-6xl mb-6 group-hover:scale-110 transition transform duration-300">{loading.extracting ? "⏳" : "📄"}</div>

            <h3 className="text-xl font-bold text-slate-700 mb-2">
              {loading.extracting ? "Reading Document..." : "Click to Upload Policy PDF/Image"}
            </h3>
            <p className="text-sm text-slate-400 mb-8">Supports: PDF, JPG, PNG</p>

            <span className="bg-blue-600 text-white px-8 py-3 rounded-full font-bold shadow-lg shadow-blue-200 group-hover:shadow-blue-300 group-hover:-translate-y-1 transition inline-block">
              Browse Files
            </span>
          </div>
        </div>

        {/* 3. Extracted Data Display (Visible in Print) */}
        {policy.company && (
          <div id="extracted-policy-card" ref={extractedDetailsRef} className="relative bg-white rounded-3xl p-8 shadow-2xl border border-slate-100 max-w-3xl mx-auto mt-6 overflow-hidden break-inside-avoid print:shadow-none print:border-none">
            {/* Decorative top accent */}


            <div className="flex items-center gap-3 mb-6 pb-2 border-b border-slate-100">
              <div className="p-2 bg-blue-50 rounded-lg text-blue-600">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" /><polyline points="14 2 14 8 20 8" /><line x1="16" y1="13" x2="8" y2="13" /><line x1="16" y1="17" x2="8" y2="17" /><polyline points="10 9 9 9 8 9" /></svg>
              </div>
              <h2 className="text-lg font-black text-slate-800 uppercase tracking-wide">Extracted Policy Details</h2>
            </div>

            <div className="space-y-4 text-slate-700">
              <div className="flex justify-between items-center group hover:bg-slate-50 p-2 rounded-lg transition-colors">
                <span className="text-xs font-bold text-slate-500 uppercase tracking-wide">Name of the Company:</span>
                <span className="text-base font-bold text-slate-900 text-right">{policy.company || "---"}</span>
              </div>

              <div className="flex justify-between items-center group hover:bg-slate-50 p-2 rounded-lg transition-colors">
                <span className="text-xs font-bold text-slate-500 uppercase tracking-wide">Name of the Product:</span>
                <span className="text-base font-bold text-slate-900 text-right">{policy.plan || "---"}</span>
              </div>

              {policy.policy_holders && policy.policy_holders.length > 0 && (
                <div className="flex justify-between items-start group hover:bg-slate-50 p-2 rounded-lg transition-colors">
                  <span className="text-xs font-bold text-slate-500 uppercase tracking-wide mt-1">Policy Holder(s):</span>
                  <div className="flex flex-col text-right">
                    {policy.policy_holders.map((person, idx) => (
                      <span key={idx} className="text-sm font-semibold text-slate-700">
                        {person.name} {person.age ? `(${person.age}yrs)` : ""}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              <div className="flex justify-between items-center group hover:bg-slate-50 p-2 rounded-lg transition-colors">
                <span className="text-xs font-bold text-slate-500 uppercase tracking-wide">Premium: </span>
                <span className="text-base font-bold text-slate-900 text-right">{policy.premium || "---"}</span>
              </div>

              <div className="bg-slate-50/50 p-4 rounded-xl border border-slate-100">
                <div className="flex justify-between items-center mb-2 pb-2 border-b border-slate-200">
                  <span className="text-xs font-bold text-slate-500 uppercase tracking-wide">Sum Insured Details</span>
                  {/* Optional: Add info icon or badge here */}
                </div>

                <div className="flex flex-col space-y-2 w-full">
                  {/* Components List */}
                  {policy.sum_insured?.components && policy.sum_insured.components.length > 0 ? (
                    policy.sum_insured.components.map((comp, idx) => (
                      <div key={idx} className="flex justify-between items-center text-sm">
                        <span className="text-slate-600 font-medium">{comp.label}:</span>
                        <span className="font-bold text-slate-900">{comp.value}</span>
                      </div>
                    ))
                  ) : (
                    /* Fallback for old/missing data */
                    policy.sum_insured?.breakdown && (
                      <div className="text-sm text-slate-500 font-medium italic mb-2 text-right">{policy.sum_insured.breakdown}</div>
                    )
                  )}

                  {/* Total */}
                  <div className="border-t border-slate-300 pt-2 mt-2 flex justify-between items-center">
                    <span className="text-sm font-bold text-slate-700">Total Sum Insured:</span>
                    <span className="text-xl font-black text-slate-900">{policy.sum_insured?.total || "---"}</span>
                  </div>
                </div>
              </div>
            </div>

            <button
              onClick={handleCompare}
              disabled={!policy.company || loading.comparing}
              className="w-full mt-8 mb-6 bg-gradient-to-r from-blue-600 to-indigo-600 text-white py-4 rounded-xl font-bold uppercase text-sm tracking-widest hover:from-blue-700 hover:to-indigo-700 hover:shadow-lg hover:shadow-indigo-200 transition transform hover:-translate-y-0.5 disabled:opacity-70 disabled:cursor-not-allowed print:hidden"
            >
              {loading.comparing ? "Analyzing Market..." : "Generate Analysis Report"}
            </button>
          </div>
        )}

        {/* SUM INSURED ANALYSIS CARD */}
        {
          policy.sum_insured?.total && report && (() => {
            const analysis = getSumInsuredAnalysis(policy.sum_insured.total);
            if (!analysis) return null;

            // [MODIFIED] Dynamic Color Theme
            const colorMap = {
              rose: { bg: "bg-rose-50", border: "border-rose-100", text: "text-rose-900", title: "text-rose-700", dot: "bg-rose-500", badge: "bg-rose-100 text-rose-700", icon: "🔴" },
              amber: { bg: "bg-amber-50", border: "border-amber-100", text: "text-amber-900", title: "text-amber-700", dot: "bg-amber-500", badge: "bg-amber-100 text-amber-700", icon: "⚠️" },
              emerald: { bg: "bg-emerald-50", border: "border-emerald-100", text: "text-emerald-900", title: "text-emerald-700", dot: "bg-emerald-500", badge: "bg-emerald-100 text-emerald-700", icon: "✅" },
              purple: { bg: "bg-purple-50", border: "border-purple-100", text: "text-purple-900", title: "text-purple-700", dot: "bg-purple-500", badge: "bg-purple-100 text-purple-700", icon: "⭐" }
            };
            const theme = colorMap[analysis.color] || colorMap.blue;

            return (
              <div ref={reportSectionRef} className={`mt-8 print:mt-4 rounded-3xl border ${theme.border} bg-white shadow-xl overflow-hidden break-inside-avoid ring-1 ring-slate-900/5`}>

                {/* Header Section */}
                <div className={`${theme.bg} px-8 py-6 border-b ${theme.border}`}>
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                    <div>
                      <div className="flex items-center gap-3 mb-2">
                        <span className="text-2xl">{theme.icon}</span>
                        <h4 className={`font-black text-2xl ${theme.title} tracking-tight`}>{analysis.status.replace(/^[^\s]+ /, "")}</h4>
                      </div>
                      <p className={`text-sm font-bold uppercase tracking-wider opacity-70 ${theme.text}`}>Current Coverage Status</p>
                    </div>
                    <div className={`px-4 py-2 rounded-xl font-bold text-sm ${theme.badge} self-start md:self-center shadow-sm`}>
                      {analysis.range_desc}
                    </div>
                  </div>
                </div>

                <div className="p-8">
                  {/* Reality Check */}
                  <div className="mb-8">
                    <div className="flex items-center gap-2 mb-4">
                      <span className="text-lg">👁️</span>
                      <h5 className="text-sm font-black text-slate-400 uppercase tracking-widest">Coverage Insights</h5>
                    </div>

                    <ul className="space-y-4">
                      {analysis.reality(policy).map((r, i) => (
                        <li key={i} className="flex gap-4 items-start text-slate-700 font-medium leading-relaxed group">
                          <div className={`mt-2 min-w-[8px] h-[8px] rounded-full ${theme.dot} ring-2 ring-white shadow-sm group-hover:scale-125 transition-transform`}></div>
                          <span className="flex-1">{r}</span>
                        </li>
                      ))}
                    </ul>
                  </div>



                  {/* Market Insight Footer - Premium Look */}
                  <div className="bg-slate-900 rounded-2xl p-6 text-white shadow-2xl relative overflow-hidden group">
                    {/* Gradient Overlay */}
                    <div className="absolute top-0 right-0 w-64 h-64 bg-gradient-to-br from-blue-600 to-purple-600 rounded-full blur-3xl opacity-20 -translate-y-1/2 translate-x-1/3 group-hover:opacity-30 transition duration-700"></div>

                    <div className="relative z-10">
                      <div className="flex items-center gap-2 mb-4 pb-4 border-b border-slate-700">
                        <span className="text-xl">💡</span>
                        <span className="font-bold text-blue-200 uppercase tracking-wider text-xs">Market Insight (2025+)</span>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm text-slate-300 mb-6">
                        <div>
                          <p className="mb-1 text-slate-400 text-xs uppercase font-bold">Avg Critical Illness Cost</p>
                          <p className="font-bold text-white text-lg mb-1">₹18–35 Lakhs</p>
                          <p className="text-xs text-slate-400 leading-relaxed font-medium">Standard 5L-10L policies often get exhausted in the first week of critical care.</p>
                        </div>
                        <div>
                          <p className="mb-1 text-slate-400 text-xs uppercase font-bold">Cancer Treatment</p>
                          <p className="font-bold text-white text-lg mb-1">₹40–50 Lakhs+</p>
                          <p className="text-xs text-slate-400 leading-relaxed font-medium">Advanced immunotherapies cost 3x more than traditional chemo.</p>
                        </div>
                        <div className="col-span-1 md:col-span-2">
                          <p className="mb-1 text-slate-400 text-xs uppercase font-bold">ICU Cost / Day</p>
                          <p className="font-medium text-slate-200 mb-1">₹50k – ₹1L+ <span className="text-slate-500 mx-2">|</span> <span className="text-blue-200 italic">₹25L+ is the new necessity.</span></p>
                          <p className="text-xs text-slate-400 leading-relaxed font-medium w-full md:w-3/4">Medical inflation is driving daily ICU charges to match luxury 5-star hotel tariffs.</p>
                        </div>
                      </div>

                      <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
                        <div className="flex items-center gap-2 mb-2">
                          <span className="text-emerald-400">✅</span>
                          <h5 className="font-bold text-emerald-400 text-sm uppercase tracking-wider">Smart Strategy</h5>
                        </div>
                        <ul className="space-y-2 text-sm text-slate-300">
                          <li className="flex justify-between border-b border-slate-700/50 pb-2 last:border-0 last:pb-0">
                            <span>Minimum</span> <span className="font-bold text-white">₹25 Lakhs</span>
                          </li>
                          <li className="flex justify-between border-b border-slate-700/50 pb-2 last:border-0 last:pb-0">
                            <span>Ideal (Base + Top-up)</span> <span className="font-bold text-white">₹50 Lakhs</span>
                          </li>
                          <li className="flex justify-between border-b border-slate-700/50 pb-2 last:border-0 last:pb-0">
                            <span>Top-up Range</span> <span className="font-bold text-white">₹50L – ₹1Cr</span>
                          </li>
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            );
          })()
        }


        {/* 4. Comparison Report (Only shows if result exists) */}
        {
          report && (
            <div className="mt-10 print:mt-4 animate-fade-in space-y-10 print:space-y-4" id="report-content">
              {/* Policy Analysis Section */}
              {/* Policy Analysis Section - Comprehensive Checklist */}
              <div className="max-w-4xl mx-auto">

                {/* Product Score Card */}
                {report.product_score && (
                  <div className="bg-gradient-to-r from-slate-900 to-slate-800 rounded-3xl p-8 text-white shadow-2xl mb-10 flex flex-col md:flex-row items-center justify-between gap-6 relative overflow-hidden">
                    <div className="absolute top-0 right-0 w-64 h-64 bg-white opacity-5 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2"></div>
                    <div>
                      <h3 className="text-2xl font-black mb-1">Policy Health Score</h3>
                      <p className="text-slate-400 text-sm">Based on 70+ parameters</p>
                    </div>
                    <div className="flex items-center gap-4 relative z-10">
                      <div className="relative w-32 h-32 flex items-center justify-center">
                        <svg className="w-full h-full transform -rotate-90">
                          <circle cx="64" cy="64" r="56" stroke="currentColor" strokeWidth="8" fill="transparent" className="text-slate-700" />
                          <circle cx="64" cy="64" r="56" stroke="currentColor" strokeWidth="8" fill="transparent" strokeDasharray={351.86} strokeDashoffset={351.86 - (351.86 * (report.product_score / 10))} className={`${report.product_score >= 7 ? 'text-emerald-500' : report.product_score >= 5 ? 'text-amber-500' : 'text-rose-500'} transition-all duration-1000`} strokeLinecap="round" />
                        </svg>
                        <div className="absolute inset-0 flex flex-col items-center justify-center">
                          <span className="text-4xl font-black">{report.product_score}</span>
                          <span className="text-xs font-bold text-slate-400 uppercase">/ 10</span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                <h3 className="text-2xl font-black text-slate-800 mb-6 text-center">Comprehensive Feature Analysis</h3>

                <div className="space-y-8">
                  {(() => {
                    const order = ["Non-Negotiable Benefits", "Must Have", "Good to Have", "Special Features"];
                    const groups = report.feature_analysis?.reduce((acc, item) => {
                      if (!acc[item.category]) acc[item.category] = [];
                      acc[item.category].push(item);
                      return acc;
                    }, {}) || {};

                    return order.map((cat) => {
                      if (!groups[cat]) return null;

                      const theme = {
                        "Non-Negotiable Benefits": "border-l-rose-500 bg-rose-50/30",
                        "Must Have": "border-l-blue-500 bg-blue-50/30",
                        "Good to Have": "border-l-emerald-500 bg-emerald-50/30",
                        "Special Features": "border-l-purple-500 bg-purple-50/30"
                      }[cat] || "border-l-slate-300";

                      return (
                        <div key={cat} className="bg-white rounded-2xl shadow-sm border border-slate-100 overflow-hidden">
                          <div className={`px-6 py-4 border-b border-slate-100 flex items-center gap-3 ${theme.split(" ")[1]}`}>
                            <div className={`w-1 h-6 rounded-full ${theme.split(" ")[0].replace("border-l-", "bg-")}`}></div>
                            <h4 className="font-bold text-lg text-slate-800">{cat}</h4>
                          </div>
                          <div className="divide-y divide-slate-50">
                            {groups[cat].map((item, idx) => (
                              <div key={idx} className="px-6 py-4 flex items-center justify-between hover:bg-slate-50 transition-colors">
                                <div className="pr-4">
                                  <p className="font-bold text-slate-700 text-sm">{item.feature}</p>
                                  <p className="text-xs text-slate-500 mt-1">{item.value}</p>
                                </div>
                                <div className="shrink-0">
                                  {item.status === "Positive" ? (
                                    <span className="w-8 h-8 rounded-full bg-emerald-100 flex items-center justify-center text-emerald-600 shadow-sm text-lg">✓</span>
                                  ) : (
                                    <span className="w-8 h-8 rounded-full bg-rose-100 flex items-center justify-center text-rose-600 shadow-sm text-sm">✕</span>
                                  )}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      );
                    });
                  })()}
                </div>
              </div>

              {/* CURRENT POLICY STATS SECTION */}
              {report.current_policy_stats && (
                <div className="max-w-4xl mx-auto break-inside-avoid print:mt-10">
                  <div className="text-center mb-6">
                    <h5 className="font-bold text-slate-500 uppercase tracking-widest text-xs">Current Insurer Performance</h5>
                    <h3 className="font-black text-2xl text-slate-800">{report.current_policy_stats.company || "Your Insurer"} Analysis</h3>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6 print:grid-cols-3 print:gap-4">
                    {/* CSR Card */}
                    <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-lg flex flex-col items-center text-center">
                      <span className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-1">Claim Settlement</span>
                      <span className="text-3xl font-black text-slate-800 mb-2">{report.current_policy_stats.csr || "N/A"}</span>
                      <span className="text-xs font-bold bg-blue-50 text-blue-600 px-3 py-1 rounded-full uppercase tracking-wide">
                        Rank: {report.current_policy_stats.csr_rank || "--"}
                      </span>
                    </div>

                    {/* Solvency Card */}
                    <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-lg flex flex-col items-center text-center">
                      <span className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-1">Solvency Ratio</span>
                      <span className="text-3xl font-black text-slate-800 mb-2">{report.current_policy_stats.solvency || "N/A"}</span>
                      <span className="text-xs font-bold bg-purple-50 text-purple-600 px-3 py-1 rounded-full uppercase tracking-wide">
                        Rank: {report.current_policy_stats.solvency_rank || "--"}
                      </span>
                    </div>

                    {/* Complaints Card */}
                    <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-lg flex flex-col items-center text-center">
                      <span className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-1">Complaints Resolved</span>
                      <span className="text-3xl font-black text-slate-800 mb-2">{report.current_policy_stats.complaints || "N/A"}</span>
                      <span className="text-xs font-bold bg-emerald-50 text-emerald-600 px-3 py-1 rounded-full uppercase tracking-wide">
                        Rank: {report.current_policy_stats.complaints_rank || "--"}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              <div className="space-y-10 print:mt-12">
                <h4 className="font-bold text-2xl text-slate-800 ml-1 border-l-8 border-blue-600 pl-4">Top Recommendations for You</h4>

                {/* Flatten all recommendations into a single list to guarantee horizontal layout */}
                {(() => {
                  const allItems = report.recommendations?.flatMap(cat => cat.items || []) || [];
                  return (
                    <div className="space-y-6 print:space-y-3">
                      {/* Use the first category name as a sub-header if available, or generic */}
                      <h5 className="font-bold text-sm uppercase tracking-wider text-slate-500 bg-slate-100 inline-block px-3 py-1 rounded-lg">
                        {report.recommendations?.[0]?.category || "Recommended Plans"}
                      </h5>

                      <div className="grid grid-cols-3 gap-6 print:gap-3">
                        {allItems.map((item, j) => (
                          <div key={j} className="bg-white p-6 rounded-2xl border border-slate-100 shadow-lg hover:shadow-2xl hover:-translate-y-1 transition-all duration-300 relative overflow-hidden group flex flex-col h-full break-inside-avoid print:p-3 print:rounded-lg">
                            <div className="absolute top-0 right-0 bg-gradient-to-l from-blue-600 to-blue-500 text-white text-[10px] font-bold px-3 py-1.5 rounded-bl-xl shadow-md opacity-90 group-hover:opacity-100 transition">
                              SHARE INDIA PARTNER
                            </div>
                            <div className="mb-4 mt-2">
                              <h6 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-1">{item.company}</h6>
                              <h5 className="font-bold text-lg text-slate-900 leading-tight mb-2">{item.name}</h5>
                              <span className="text-[10px] font-bold bg-blue-50 text-blue-600 px-2 py-1 rounded border border-blue-100 uppercase tracking-wide">{item.type}</span>
                            </div>


                            <p className="text-sm text-slate-600 leading-relaxed mb-4 border-b border-slate-50 pb-4 print:text-xs print:mb-2 print:pb-2">{item.description}</p>

                            {/* Product Score Display */}
                            {item.product_score && (
                              <div className="flex items-center gap-2 mb-4 bg-slate-50 p-2 rounded-lg border border-slate-100 w-fit">
                                <div className="relative w-8 h-8 flex items-center justify-center">
                                  <svg className="w-full h-full transform -rotate-90">
                                    <circle cx="16" cy="16" r="14" stroke="currentColor" strokeWidth="3" fill="transparent" className="text-slate-200" />
                                    <circle cx="16" cy="16" r="14" stroke="currentColor" strokeWidth="3" fill="transparent" strokeDasharray={87.96} strokeDashoffset={87.96 - (87.96 * (item.product_score / 10))} className={`${item.product_score >= 8 ? 'text-emerald-500' : 'text-blue-500'}`} strokeLinecap="round" />
                                  </svg>
                                  <span className="absolute text-[9px] font-bold text-slate-700">{item.product_score}</span>
                                </div>
                                <span className="text-[10px] font-bold text-slate-500 uppercase tracking-tight">Product Score</span>
                              </div>
                            )}

                            <div className="mt-4 mb-4 grid grid-cols-3 gap-2">
                              <div className="bg-slate-50 p-2 rounded-xl border border-slate-100 text-center print:p-1">
                                <span className="block text-[8px] font-bold text-slate-400 uppercase tracking-widest text-nowrap">Claims Paid</span>
                                <span className="block text-sm font-black text-slate-700 print:text-xs">{item.stats?.csr || "N/A"}</span>
                                <span className="block text-[9px] font-bold text-blue-600 mt-1 uppercase tracking-wide">Rank: {item.stats?.csr_rank || "-"}</span>
                              </div>
                              <div className="bg-slate-50 p-2 rounded-xl border border-slate-100 text-center print:p-1">
                                <span className="block text-[8px] font-bold text-slate-400 uppercase tracking-widest text-nowrap">Solvency</span>
                                <span className="block text-sm font-black text-slate-700 print:text-xs">{item.stats?.solvency || "N/A"}</span>
                                <span className="block text-[9px] font-bold text-purple-600 mt-1 uppercase tracking-wide">Rank: {item.stats?.solvency_rank || "-"}</span>
                              </div>
                              <div className="bg-slate-50 p-2 rounded-xl border border-slate-100 text-center print:p-1">
                                <span className="block text-[8px] font-bold text-slate-400 uppercase tracking-widest text-nowrap">Complaints</span>
                                <span className="block text-sm font-black text-slate-700 print:text-xs">{item.stats?.complaints || "N/A"}</span>
                                <span className="block text-[9px] font-bold text-emerald-600 mt-1 uppercase tracking-wide">Rank: {item.stats?.complaints_rank || "-"}</span>
                              </div>
                            </div>

                            <div className="mt-4 mb-6 print:mt-2 print:mb-2">
                              <span className="text-xs font-bold text-emerald-600 block mb-2 uppercase tracking-wide">✅ BENEFITS</span>
                              <ul className="text-sm space-y-2 text-slate-600 print:text-xs print:space-y-1">
                                {item.benefits?.map((b, k) => <li key={k} className="leading-snug flex gap-1.5"><span className="text-emerald-500">•</span> {b}</li>)}
                              </ul>
                            </div>

                            {/* [NEW] Compare Button */}
                            <button
                              onClick={() => setComparingItem(item)}
                              className="w-full mt-auto bg-white border-2 border-slate-900 text-slate-900 py-2 rounded-xl font-bold uppercase text-xs tracking-widest hover:bg-slate-900 hover:text-white transition print:py-1 print:text-[10px]"
                            >
                              Compare
                            </button>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })()}
              </div>

              {/* [MOVED] Action Buttons - Bottom */}
              <div className="flex justify-center print:hidden pb-10">
                <button
                  onClick={() => window.print()}
                  className="bg-indigo-600 text-white px-6 py-3 rounded-lg font-bold text-sm uppercase tracking-widest hover:bg-indigo-700 transition flex items-center gap-2 shadow-lg hover:-translate-y-0.5 transform"
                >
                  <span className="text-lg">🖨️</span> Save as PDF / Print
                </button>
              </div>
            </div>
          )
        }
      </main >

      {/* [NEW] Comparison Modal */}
      {
        comparingItem && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/90 backdrop-blur-md animate-fade-in print:absolute print:bg-white print:p-0 print:block">
            <div className="bg-white rounded-3xl w-full max-w-7xl h-[90vh] flex flex-col shadow-2xl relative overflow-hidden print:h-auto print:max-h-none print:w-full print:rounded-none print:shadow-none print:border-none print:overflow-visible">
              {/* Modal Header - Compact */}
              <div className="px-6 py-3 border-b flex justify-between items-center bg-slate-50 shrink-0">
                <h3 className="text-lg font-black text-slate-900">Policy Comparison</h3>
                <button
                  onClick={() => setComparingItem(null)}
                  className="w-8 h-8 rounded-full bg-slate-200 hover:bg-slate-300 flex items-center justify-center transition text-sm text-slate-600"
                >
                  ✕
                </button>
              </div>

              {/* Combined Scrollable Content */}
              <div className="flex-1 overflow-auto print:overflow-visible print:h-auto print:block">
                <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-0 md:gap-8 divide-y md:divide-y-0 md:divide-x divide-slate-100">
                  {/* Left Column: Your Policy */}
                  <div className="space-y-4 pl-4">
                    <div className="bg-gray-100 text-gray-600 text-[10px] font-extrabold px-3 py-1 rounded inline-block uppercase tracking-wider mb-2">
                      Existing Policy
                    </div>

                    <div className="flex flex-col gap-3">
                      <div className="flex items-baseline gap-2">
                        <span className="text-sm font-bold text-slate-400 uppercase tracking-wide min-w-[80px]">Company:</span>
                        <span className="text-lg font-black text-slate-800 break-words line-clamp-2 leading-tight">
                          {policy.company ? policy.company.replace(/((?:Co\.|Corp\.|Ltd\.|Limited)\.?)(.*)/i, '$1').trim() : "Unknown"}
                        </span>
                      </div>
                      <div className="flex items-baseline gap-2">
                        <span className="text-sm font-bold text-slate-400 uppercase tracking-wide min-w-[80px]">Plan:</span>
                        <span className="text-lg font-bold text-slate-700">{policy.plan || "N/A"}</span>
                      </div>
                      <div className="flex items-baseline gap-2">
                        <span className="text-sm font-bold text-slate-400 uppercase tracking-wide min-w-[80px]">Premium:</span>
                        <span className="text-lg font-bold text-slate-700">{policy.premium || "N/A"}</span>
                      </div>
                    </div>
                  </div>

                  {/* Right Column: Recommended Policy */}
                  <div className="space-y-4 md:pl-8 pt-6 md:pt-0">
                    <div className="bg-blue-100 text-blue-700 text-[10px] font-extrabold px-3 py-1 rounded inline-block uppercase tracking-wider mb-2">
                      Recommended Upgrade
                    </div>

                    <div className="flex flex-col gap-3">
                      <div className="flex items-baseline gap-2">
                        <span className="text-sm font-bold text-blue-300 uppercase tracking-wide min-w-[80px]">Company:</span>
                        <span className="text-lg font-black text-blue-800">{comparingItem.company || "Unknown"}</span>
                      </div>
                      <div className="flex items-baseline gap-2">
                        <span className="text-sm font-bold text-blue-300 uppercase tracking-wide min-w-[80px]">Plan:</span>
                        <div className="flex-1">
                          <span className="text-xl font-black text-blue-700 leading-tight">{comparingItem.name}</span>
                          <span className="text-xl font-bold text-blue-600 ml-2">({comparingItem.type})</span>
                        </div>
                      </div>

                      <div className="flex items-baseline gap-2">
                        <span className="text-sm font-bold text-blue-300 uppercase tracking-wide min-w-[80px]">Est. Cost:</span>
                        <span className="text-2xl font-black text-transparent bg-clip-text bg-gradient-to-r from-blue-700 to-indigo-600 print:text-blue-700 print:bg-none">
                          {comparingItem.premium}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Detailed Comparison Table - Full Width Centered */}
                <div className="px-6 pb-6">
                  <div className="max-w-5xl mx-auto">
                    <div className="text-center mb-6">
                      <div className="inline-flex items-center gap-2 bg-blue-50 px-3 py-1.5 rounded-full border border-blue-100">
                        <span className="text-lg">📊</span>
                        <span className="text-xs text-blue-800 font-black uppercase tracking-widest">Detailed Gap Analysis</span>
                      </div>
                    </div>

                    <div className="space-y-8">
                      {/* Helper to render a category section */}
                      {['non_negotiable', 'must_have', 'good_to_have', 'special_features'].map((key) => {
                        const items = comparingItem[key];
                        if (!items || items.length === 0) return null;

                        const titles = {
                          non_negotiable: "Non-Negotiable Benefits",
                          must_have: "Must Have Features",
                          good_to_have: "Good To Have Features",
                          special_features: "Special Features"
                        };

                        const colors = {
                          non_negotiable: "indigo",
                          must_have: "blue",
                          good_to_have: "purple",
                          special_features: "amber"
                        };

                        const color = colors[key];

                        return (
                          <div key={key} className="border border-slate-200 rounded-xl overflow-hidden shadow-sm">
                            <div className={`bg-${color}-50 px-6 py-3 border-b border-${color}-100 flex items-center gap-2`}>
                              {key === 'non_negotiable' && <span className="text-lg">💎</span>}
                              <h5 className={`font-black text-${color}-800 text-sm uppercase tracking-wider`}>{titles[key]}</h5>
                            </div>

                            {/* Table Header */}
                            <div className="grid grid-cols-12 bg-slate-50 border-b border-slate-200 text-[10px] font-black text-slate-400 uppercase tracking-wider text-center">
                              <div className="col-span-4 p-2">Feature</div>
                              <div className="col-span-4 p-2 border-l border-slate-200">Your Current Policy</div>
                              <div className="col-span-4 p-2 border-l border-slate-200 bg-blue-50/30 text-blue-600">Recommended Upgrade</div>
                            </div>

                            <div className="divide-y divide-slate-100 bg-white">
                              {items
                                .filter(row => {
                                  const e = (row.existing || "").trim().toLowerCase();
                                  const p = (row.proposed || "").trim().toLowerCase();

                                  // [NEW] Strictly hide if recommended data is missing from DB
                                  if (p === "n/a" || p.includes("no data") || p.includes("no") || p.includes("ref 3") || p.includes("unavailable") || p.includes("not listed")) return false;

                                  // aggressive check for "bad" feature values
                                  const isBad = (val) => {
                                    if (!val) return true;
                                    if (["no", "unknown", "n/a", "not mentioned", "not found", "not listed"].includes(val)) return true;
                                    if (val.includes("no/") || val.includes("/no")) return true; // catches "No/Not Listed"
                                    if (val.startsWith("no ")) return true; // catches "No coverage"
                                    return false;
                                  };

                                  const existingIsBad = isBad(e) || e.includes("likely capped") || e.includes("limited");
                                  const proposedIsBad = isBad(p);

                                  // remove if BOTH are bad
                                  return !(existingIsBad && proposedIsBad);
                                })
                                .map((row, i) => (
                                  <div key={i} className="grid grid-cols-12 items-stretch text-xs">
                                    <div className="col-span-4 p-3 bg-slate-50/30 flex items-center justify-center text-center font-bold text-slate-700 break-words">
                                      {row.feature}
                                    </div>
                                    <div className="col-span-4 p-3 border-l border-slate-100 flex items-center justify-center text-center text-slate-500 font-medium break-words">
                                      {(row.existing && row.existing.toLowerCase() === "not found") ? "Not Available" : (row.existing || <span className="italic text-slate-300">--</span>)}
                                    </div>
                                    <div className="col-span-4 p-3 border-l border-blue-50 bg-blue-50/10 flex items-center justify-center text-center font-bold text-blue-800 relative break-words">
                                      {row.proposed}
                                      {row.status === "Upgrade" && (
                                        <span className="absolute top-1 right-1 text-[8px] bg-emerald-100 text-emerald-700 px-1 rounded font-bold uppercase tracking-wide">UPGRADE</span>
                                      )}
                                    </div>
                                  </div>
                                ))}
                            </div>
                          </div>
                        );
                      })}

                      {/* RED FLAGS SECTION */}
                      {comparingItem.red_flags && comparingItem.red_flags.length > 0 && (
                        <div className="bg-rose-50 border border-rose-100 rounded-xl p-6">
                          <h5 className="font-bold text-rose-800 text-sm uppercase tracking-wider mb-4 flex items-center gap-2">
                            <span>🚩</span> Red Flags / Things to Avoid
                          </h5>
                          <ul className="space-y-2">
                            {comparingItem.red_flags.map((flag, i) => (
                              <li key={i} className="flex gap-2 text-rose-900 text-sm font-medium">
                                <span className="text-rose-500 font-bold">•</span> {flag}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {/* Fallback if nothing exists */}
                      {(!comparingItem.must_have && !comparingItem.good_to_have && !comparingItem.special_features && !comparingItem.comparison_table) && (
                        <div className="p-8 text-center bg-slate-50 rounded-xl border border-dashed border-slate-300">
                          <p className="text-slate-400 text-sm">Detailed comparison data unavailable for this item.</p>
                        </div>
                      )}
                    </div>

                    <div className="pt-6">
                      <p className="text-xs text-slate-500 italic bg-slate-50 p-3 rounded-xl border border-slate-100 text-center">
                        "{comparingItem.description}"
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Modal Footer - Compact */}
              <div className="px-6 py-3 border-t bg-slate-50 flex justify-between items-center shrink-0">
                <button
                  onClick={() => window.print()}
                  className="text-slate-500 hover:text-slate-800 text-sm font-bold uppercase tracking-wide flex items-center gap-2"
                >
                  🖨️ Print / Save PDF
                </button>
                <button
                  onClick={() => setComparingItem(null)}
                  className="bg-slate-900 hover:bg-slate-800 text-white px-6 py-2 rounded-lg font-bold text-sm transition"
                >
                  Close Comparison
                </button>
              </div>
            </div>
          </div>
        )
      }


    </div >
  );
}
